import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
from datetime import datetime
from html import unescape

import pandas as pd
import requests
import yfinance as yf
from llama_index.core import (Settings, Document, VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv


load_dotenv("config.env")

DEFAULT_STOCK_DB_PATH = os.getenv("STOCK_SQL_DB_PATH")
DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR")
DEFAULT_STOCK_FILINGS_BASE_DIR = os.getenv("STOCK_FILINGS_BASE_DIR", "./data_store/filings")
RELATIONSHIP_GRAPH_FILENAME = "relationship_graph.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE = (
        "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dashes}/{primary_document}"
    )
DEFAULT_SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
_SEC_TICKER_MAP = None

def env():
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    load_dotenv("config.env")
    Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_API_KEY'))

"""
    ingestion and updating process

    1. update_financial_records(sql)
        -> create or update the record of all quarterly and annuelly record of financial indicators in sql

    2. build_financial_docs(doc file) -> create llamaindex doc file

    3. refresh_ticker_data_and_index(vector store)
        -> which includes previous two steps, and parse the doc file and save it into vector index

    basically, I can just run refresh_ticker_data_and_index(), which does everything
    I can also only run update_financial_records() which only updates sql_db.
"""


def _clean_profile_field(value):
    if value is None:
        return None
    if pd.isna(value):
        return None

    cleaned_value = str(value).strip()
    return cleaned_value or None


def _sec_headers(accept="text/html,application/json;q=0.9,*/*;q=0.8"):
    return {
        "User-Agent": DEFAULT_SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": accept,
        "Connection": "keep-alive",
    }


def _sec_get_json(url, timeout=30):
    response = requests.get(
        url,
        headers=_sec_headers(accept="application/json,text/plain,*/*"),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _sec_get_text(url, timeout=30):
    response = requests.get(url, headers=_sec_headers(), timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def _load_sec_ticker_map():
    global _SEC_TICKER_MAP
    if _SEC_TICKER_MAP is not None:
        return _SEC_TICKER_MAP

    ticker_map = {}
    for item in _sec_get_json(SEC_TICKERS_URL).values():
        ticker = (item.get("ticker") or "").upper()
        cik = str(item.get("cik_str") or "").strip()
        title = _clean_profile_field(item.get("title"))
        if ticker and cik:
            ticker_map[ticker] = {"cik": cik.zfill(10), "title": title}

    _SEC_TICKER_MAP = ticker_map
    return _SEC_TICKER_MAP


def _truncate_text(text, max_chars=2400):
    if not text:
        return ""
    trimmed = re.sub(r"\s+", " ", text).strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3].rstrip() + "..."


def _derive_sector_from_sic(sic_code, sic_description):
    description = (sic_description or "").lower()
    code_text = str(sic_code or "").strip()

    if any(keyword in description for keyword in [
        "beverage", "food", "retail", "consumer", "restaurant", "apparel", "household", "cosmetic"
    ]):
        return "Consumer Defensive"
    if any(keyword in description for keyword in [
        "software", "semiconductor", "computer", "communications", "internet", "data processing"
    ]):
        return "Technology"
    if any(keyword in description for keyword in [
        "pharmaceutical", "biotech", "medical", "health", "hospital"
    ]):
        return "Healthcare"
    if any(keyword in description for keyword in [
        "bank", "insurance", "financial", "asset", "investment", "capital"
    ]):
        return "Financial Services"
    if any(keyword in description for keyword in [
        "oil", "gas", "energy", "pipeline", "coal", "electric"
    ]):
        return "Energy"
    if any(keyword in description for keyword in [
        "industrial", "machinery", "transportation", "aerospace", "manufacturing", "construction"
    ]):
        return "Industrials"
    if any(keyword in description for keyword in [
        "telecom", "media", "entertainment", "broadcasting"
    ]):
        return "Communication Services"
    if any(keyword in description for keyword in ["real estate", "reit", "property"]):
        return "Real Estate"
    if code_text.startswith(("48", "49")):
        return "Utilities"
    return None


def _html_to_text_preserve_lines(html):
    if not html:
        return ""

    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"(?i)</div\s*>", "\n", text)
    text = re.sub(r"(?i)</tr\s*>", "\n", text)
    text = re.sub(r"(?i)</li\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_section_text(text):
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^[ \t]+", "", text)
    return text.strip()


def _find_best_section(text, start_patterns, end_patterns, min_length=1200):
    if not text:
        return ""

    matches = []
    for start_pattern in start_patterns:
        for match in re.finditer(start_pattern, text, flags=re.IGNORECASE):
            start_index = match.start()
            end_index = len(text)
            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, text[match.end():], flags=re.IGNORECASE)
                if end_match:
                    candidate_end = match.end() + end_match.start()
                    if candidate_end > start_index:
                        end_index = min(end_index, candidate_end)
            section = _clean_section_text(text[start_index:end_index])
            if len(section) >= min_length:
                matches.append((start_index, section))

    if not matches:
        return ""
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _split_paragraphs(text):
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    return [re.sub(r"\s+", " ", block).strip() for block in blocks if re.sub(r"\s+", " ", block).strip()]


def _build_keyword_paragraph_snippet(text, keywords, max_paragraphs=6):
    if not text:
        return ""

    normalized_keywords = [keyword.lower() for keyword in keywords]
    matching_blocks = []
    for block in _split_paragraphs(text):
        lowered_block = block.lower()
        if any(keyword in lowered_block for keyword in normalized_keywords):
            matching_blocks.append(block)
        if len(matching_blocks) >= max_paragraphs:
            break

    return "\n\n".join(matching_blocks)


def _clear_windows_readonly(path):
    if os.name != "nt" or not os.path.lexists(path):
        return
    subprocess.run(
        ["cmd", "/c", "attrib", "-R", path],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _handle_remove_readonly(func, path, exc_info):
    _clear_windows_readonly(path)
    func(path)


def _is_windows_reparse_point(path):
    if os.name != "nt" or not os.path.lexists(path):
        return False
    try:
        return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def _reset_persist_dir(persist_dir):
    if os.path.lexists(persist_dir):
        _clear_windows_readonly(persist_dir)
        if _is_windows_reparse_point(persist_dir):
            os.rmdir(persist_dir)
        else:
            shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
    os.makedirs(persist_dir, exist_ok=True)


def _build_recent_10k_sections(ticker, max_filings=2):
    ticker = ticker.upper()
    ticker_map = _load_sec_ticker_map()
    sec_identity = ticker_map.get(ticker)
    if not sec_identity:
        raise ValueError(f"Ticker {ticker} was not found in the SEC ticker map.")

    cik = sec_identity["cik"]
    submissions = _sec_get_json(SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    recent = ((submissions.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_documents = recent.get("primaryDocument") or []

    filings = []
    for form, accession_number, filing_date, primary_document in zip(
        forms,
        accession_numbers,
        filing_dates,
        primary_documents,
    ):
        if form != "10-K":
            continue
        filings.append(
            {
                "accession_number": accession_number,
                "filing_date": filing_date,
                "primary_document": primary_document,
            }
        )
        if len(filings) >= max_filings:
            break

    recent_sections = []
    cik_no_zero = str(int(cik))
    sic_code = _clean_profile_field(submissions.get("sic"))
    sic_description = _clean_profile_field(submissions.get("sicDescription"))
    company_name = _clean_profile_field(submissions.get("name")) or sec_identity.get("title") or ticker

    for filing in filings:
        accession_no_dashes = filing["accession_number"].replace("-", "")
        filing_url = SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE.format(
            cik_no_zero=cik_no_zero,
            accession_no_dashes=accession_no_dashes,
            primary_document=filing["primary_document"],
        )
        filing_html = _sec_get_text(filing_url)
        filing_text = _html_to_text_preserve_lines(filing_html)

        item_1_text = _find_best_section(
            filing_text,
            start_patterns=[
                r"\bitem\s+1[\.\s:;-]*business\b",
                r"\bitem\s+1[\.\s:;-]*overview\b",
            ],
            end_patterns=[
                r"\bitem\s+1a[\.\s:;-]*risk factors\b",
                r"\bitem\s+2[\.\s:;-]*properties\b",
                r"\bitem\s+2[\.\s:;-]*facilities\b",
            ],
            min_length=1200,
        )
        item_1a_text = _find_best_section(
            filing_text,
            start_patterns=[r"\bitem\s+1a[\.\s:;-]*risk factors\b"],
            end_patterns=[
                r"\bitem\s+1b[\.\s:;-]*unresolved staff comments\b",
                r"\bitem\s+2[\.\s:;-]*properties\b",
                r"\bitem\s+2[\.\s:;-]*facilities\b",
            ],
            min_length=1200,
        )

        recent_sections.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "cik": cik,
                "sic": sic_code,
                "sic_description": sic_description,
                "filing_date": filing["filing_date"],
                "filing_url": filing_url,
                "item_1_text": item_1_text,
                "item_1_excerpt": _truncate_text(item_1_text, max_chars=3200),
                "item_1a_text": item_1a_text,
                "item_1a_excerpt": _truncate_text(item_1a_text, max_chars=2200),
            }
        )

    return recent_sections


def _get_yahoo_taxonomy_fallback(ticker):
    stock = yf.Ticker(ticker.upper())
    try:
        info = stock.get_info()
    except Exception:
        info = {}
    return {
        "company_name": (
            _clean_profile_field(info.get("longName"))
            or _clean_profile_field(info.get("shortName"))
        ),
        "sector": _clean_profile_field(info.get("sector")),
        "industry": _clean_profile_field(info.get("industry")),
    }


def get_company_profile(ticker):
    ticker = ticker.upper()

    try:
        recent_10k_sections = _build_recent_10k_sections(ticker, max_filings=2)
    except Exception as exc:
        print(f"Skipping SEC company profile fetch for {ticker}: {exc}")
        recent_10k_sections = []

    latest_filing = recent_10k_sections[0] if recent_10k_sections else {}
    yahoo_fallback = _get_yahoo_taxonomy_fallback(ticker)

    company_name = (
        latest_filing.get("company_name")
        or yahoo_fallback.get("company_name")
        or ticker
    )
    industry = latest_filing.get("sic_description") or yahoo_fallback.get("industry")
    sector = _derive_sector_from_sic(
        latest_filing.get("sic"),
        latest_filing.get("sic_description"),
    ) or yahoo_fallback.get("sector")
    description = latest_filing.get("item_1_excerpt") or ""
    source = "SEC EDGAR 10-K Item 1 Business" if recent_10k_sections else "Yahoo Finance via yfinance"

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "description": description,
        "source": source,
        "cik": latest_filing.get("cik"),
        "sic": latest_filing.get("sic"),
        "sic_description": latest_filing.get("sic_description"),
        "latest_filing_date": latest_filing.get("filing_date"),
        "latest_filing_url": latest_filing.get("filing_url"),
        "recent_10k_sections": recent_10k_sections,
    }


def _filing_form_dir(ticker, form_type, filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR):
    return os.path.join(
        filings_base_dir,
        ticker.upper(),
        form_type.lower().replace("-", ""),
    )


def _filing_json_name(filing_date, accession_number):
    return f"{filing_date}_{accession_number.replace('-', '')}.json"


def _extract_10k_sections(filing_text):
    return {
        "item_1_business": _find_best_section(
            filing_text,
            start_patterns=[
                r"\bitem\s+1[\.\s:;-]*business\b",
                r"\bitem\s+1[\.\s:;-]*overview\b",
            ],
            end_patterns=[
                r"\bitem\s+1a[\.\s:;-]*risk factors\b",
                r"\bitem\s+2[\.\s:;-]*properties\b",
                r"\bitem\s+2[\.\s:;-]*facilities\b",
            ],
            min_length=1200,
        ),
        "item_1a_risk_factors": _find_best_section(
            filing_text,
            start_patterns=[r"\bitem\s+1a[\.\s:;-]*risk factors\b"],
            end_patterns=[
                r"\bitem\s+1b[\.\s:;-]*unresolved staff comments\b",
                r"\bitem\s+2[\.\s:;-]*properties\b",
                r"\bitem\s+2[\.\s:;-]*facilities\b",
            ],
            min_length=1200,
        ),
        "item_7_mda": _find_best_section(
            filing_text,
            start_patterns=[
                r"\bitem\s+7[\.\s:;-]*management'?s discussion and analysis of financial condition and results of operations\b",
                r"\bitem\s+7[\.\s:;-]*management'?s discussion and analysis\b",
            ],
            end_patterns=[
                r"\bitem\s+7a[\.\s:;-]*quantitative and qualitative disclosures about market risk\b",
                r"\bitem\s+8[\.\s:;-]*financial statements",
            ],
            min_length=1600,
        ),
        "item_8_financial_statements": _find_best_section(
            filing_text,
            start_patterns=[r"\bitem\s+8[\.\s:;-]*financial statements(?: and supplementary data)?\b"],
            end_patterns=[
                r"\bitem\s+9[\.\s:;-]*changes in and disagreements with accountants\b",
                r"\bitem\s+9a[\.\s:;-]*controls and procedures\b",
            ],
            min_length=1800,
        ),
    }


def _extract_10q_sections(filing_text):
    return {
        "item_1_financial_statements": _find_best_section(
            filing_text,
            start_patterns=[r"\bitem\s+1[\.\s:;-]*financial statements\b"],
            end_patterns=[r"\bitem\s+2[\.\s:;-]*management'?s discussion and analysis\b"],
            min_length=1200,
        ),
        "item_2_mda": _find_best_section(
            filing_text,
            start_patterns=[
                r"\bitem\s+2[\.\s:;-]*management'?s discussion and analysis of financial condition and results of operations\b",
                r"\bitem\s+2[\.\s:;-]*management'?s discussion and analysis\b",
            ],
            end_patterns=[
                r"\bitem\s+3[\.\s:;-]*quantitative and qualitative disclosures about market risk\b",
                r"\bitem\s+4[\.\s:;-]*controls and procedures\b",
                r"\bitem\s+1a[\.\s:;-]*risk factors\b",
            ],
            min_length=1400,
        ),
        "item_1a_risk_factors": _find_best_section(
            filing_text,
            start_patterns=[r"\bitem\s+1a[\.\s:;-]*risk factors\b"],
            end_patterns=[
                r"\bitem\s+2[\.\s:;-]*unregistered sales of equity securities\b",
                r"\bitem\s+5[\.\s:;-]*other information\b",
                r"\bitem\s+6[\.\s:;-]*exhibits\b",
            ],
            min_length=1000,
        ),
    }


def _document_payloads_from_section_map(ticker, company_profile, filing_record):
    form_type = filing_record["form_type"]
    section_map = filing_record["sections"]
    payloads = []

    if form_type == "10-K":
        ordered_sections = [
            ("item_1_business", "SEC 10-K Item 1 Business"),
            ("item_1a_risk_factors", "SEC 10-K Item 1A Risk Factors"),
            ("item_7_mda", "SEC 10-K Item 7 MD&A"),
            ("item_8_financial_statements", "SEC 10-K Item 8 Financial Statements"),
        ]
    else:
        ordered_sections = [
            ("item_1_financial_statements", "SEC 10-Q Item 1 Financial Statements"),
            ("item_2_mda", "SEC 10-Q Item 2 MD&A"),
            ("item_1a_risk_factors", "SEC 10-Q Item 1A Risk Factors"),
        ]

    for section_key, section_title in ordered_sections:
        section_text = section_map.get(section_key) or ""
        if not section_text:
            continue

        payloads.append(
            {
                "text": (
                    f"**{section_title}**\n\n"
                    f"Company: {company_profile['company_name']}\n"
                    f"Ticker: {ticker}\n"
                    f"Form Type: {form_type}\n"
                    f"Filing Date: {filing_record['filing_date']}\n"
                    f"Filing URL: {filing_record['filing_url']}\n"
                    f"Sector: {company_profile['sector'] or 'Unknown'}\n"
                    f"Industry: {company_profile['industry'] or 'Unknown'}\n\n"
                    f"{section_text}"
                ),
                "metadata": {
                    "ticker": ticker,
                    "type": "sec_filing_section",
                    "form_type": form_type,
                    "section_key": section_key,
                    "section_title": section_title,
                    "company_name": company_profile["company_name"],
                    "sector": company_profile["sector"] or "Unknown",
                    "industry": company_profile["industry"] or "Unknown",
                    "filing_date": filing_record["filing_date"],
                    "filing_url": filing_record["filing_url"],
                    "accession_number": filing_record["accession_number"],
                    "source": section_title,
                },
            }
        )

    return payloads


def _persist_filing_payloads(ticker, form_type, filing_records, filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR):
    form_dir = _filing_form_dir(ticker, form_type, filings_base_dir=filings_base_dir)
    os.makedirs(form_dir, exist_ok=True)

    retained_files = set()
    for filing_record in filing_records:
        file_name = _filing_json_name(
            filing_record["filing_date"],
            filing_record["accession_number"],
        )
        retained_files.add(file_name)
        file_path = os.path.join(form_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as filing_file:
            json.dump(filing_record, filing_file, ensure_ascii=False, indent=2)

    for existing_name in os.listdir(form_dir):
        existing_path = os.path.join(form_dir, existing_name)
        if existing_name.endswith(".json") and existing_name not in retained_files:
            os.remove(existing_path)


def _documents_from_payloads(document_payloads):
    return [
        Document(text=payload["text"], metadata=payload["metadata"])
        for payload in document_payloads
    ]


def _build_company_profile_from_sec_archive(ticker, archive_payload):
    latest_10k = (archive_payload.get("10k_filings") or [{}])[0]
    yahoo_fallback = _get_yahoo_taxonomy_fallback(ticker)

    company_name = (
        latest_10k.get("company_name")
        or archive_payload.get("company_name")
        or yahoo_fallback.get("company_name")
        or ticker
    )
    industry = latest_10k.get("sic_description") or archive_payload.get("sic_description") or yahoo_fallback.get("industry")
    sector = _derive_sector_from_sic(
        latest_10k.get("sic"),
        latest_10k.get("sic_description"),
    ) or yahoo_fallback.get("sector")

    item_1_text = (latest_10k.get("sections") or {}).get("item_1_business") or ""
    description = _truncate_text(item_1_text, max_chars=3200)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "description": description,
        "source": "SEC EDGAR filings archive" if latest_10k else "Yahoo Finance via yfinance",
        "cik": archive_payload.get("cik"),
        "sic": latest_10k.get("sic") or archive_payload.get("sic"),
        "sic_description": latest_10k.get("sic_description") or archive_payload.get("sic_description"),
        "latest_filing_date": latest_10k.get("filing_date"),
        "latest_filing_url": latest_10k.get("filing_url"),
        "ten_k_filings": archive_payload.get("10k_filings", []),
        "ten_q_filings": archive_payload.get("10q_filings", []),
    }


def _sync_sec_filing_archive(
    ticker,
    filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR,
    max_10k_filings=10,
    max_10q_filings=12):
    ticker = ticker.upper()
    ticker_map = _load_sec_ticker_map()
    sec_identity = ticker_map.get(ticker)
    if not sec_identity:
        raise ValueError(f"Ticker {ticker} was not found in the SEC ticker map.")

    cik = sec_identity["cik"]
    submissions = _sec_get_json(SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    recent = ((submissions.get("filings") or {}).get("recent") or {})
    company_name = _clean_profile_field(submissions.get("name")) or sec_identity.get("title") or ticker
    sic = _clean_profile_field(submissions.get("sic"))
    sic_description = _clean_profile_field(submissions.get("sicDescription"))

    base_profile = {
        "company_name": company_name,
        "sector": _derive_sector_from_sic(sic, sic_description),
        "industry": sic_description,
    }

    filing_groups = {"10-K": [], "10-Q": []}
    max_by_form = {"10-K": max_10k_filings, "10-Q": max_10q_filings}
    cik_no_zero = str(int(cik))

    for form, accession_number, filing_date, primary_document in zip(
        recent.get("form") or [],
        recent.get("accessionNumber") or [],
        recent.get("filingDate") or [],
        recent.get("primaryDocument") or [],
    ):
        if form not in filing_groups or len(filing_groups[form]) >= max_by_form[form]:
            continue

        accession_no_dashes = accession_number.replace("-", "")
        filing_url = SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE.format(
            cik_no_zero=cik_no_zero,
            accession_no_dashes=accession_no_dashes,
            primary_document=primary_document,
        )
        filing_html = _sec_get_text(filing_url)
        filing_text = _html_to_text_preserve_lines(filing_html)
        sections = _extract_10k_sections(filing_text) if form == "10-K" else _extract_10q_sections(filing_text)

        filing_record = {
            "ticker": ticker,
            "company_name": company_name,
            "cik": cik,
            "sic": sic,
            "sic_description": sic_description,
            "form_type": form,
            "filing_date": filing_date,
            "filing_url": filing_url,
            "accession_number": accession_number,
            "primary_document": primary_document,
            "sections": sections,
        }
        filing_record["documents"] = _document_payloads_from_section_map(
            ticker,
            base_profile | {"company_name": company_name},
            filing_record,
        )
        filing_groups[form].append(filing_record)

    _persist_filing_payloads(ticker, "10-K", filing_groups["10-K"], filings_base_dir=filings_base_dir)
    _persist_filing_payloads(ticker, "10-Q", filing_groups["10-Q"], filings_base_dir=filings_base_dir)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "cik": cik,
        "sic": sic,
        "sic_description": sic_description,
        "10k_filings": filing_groups["10-K"],
        "10q_filings": filing_groups["10-Q"],}


def _empty_relationship_graph(ticker, company_profile):
    ticker = ticker.upper()
    company_name = company_profile.get("company_name") or ticker

    return {
        "ticker": ticker,
        "company_name": company_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sources": [],
        "nodes": [
            {
                "id": f"company::{ticker}",
                "name": company_name,
                "entity_type": "company",
                "roles": ["company"],
                "profile": company_profile.get("description") or "",
                "aliases": [ticker, company_name],
                "source_labels": ["company_profile"],
            }
        ],
        "edges": [],
    }


def _normalize_entity_key(name):
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _extract_json_payload(raw_text):
    if not raw_text:
        return {}

    cleaned_text = raw_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        start_index = cleaned_text.find("{")
        end_index = cleaned_text.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            return {}
        try:
            return json.loads(cleaned_text[start_index:end_index + 1])
        except json.JSONDecodeError:
            return {}


def _build_relationship_source_snippets(ticker, company_profile):
    sources = []

    profile_lines = [
        f"Company: {company_profile['company_name']}",
        f"Ticker: {ticker}",
        f"Sector: {company_profile['sector'] or 'Unknown'}",
        f"Industry: {company_profile['industry'] or 'Unknown'}",
        f"Source: {company_profile['source']}",
    ]
    if company_profile.get("latest_filing_date"):
        profile_lines.append(f"Latest 10-K Filing Date: {company_profile['latest_filing_date']}")
    if company_profile.get("latest_filing_url"):
        profile_lines.append(f"Latest 10-K Filing URL: {company_profile['latest_filing_url']}")
    if company_profile.get("description"):
        profile_lines.append(f"Business Summary Excerpt: {company_profile['description']}")

    sources.append(
        {
            "label": "company_profile",
            "source_type": "sec_company_profile",
            "text": "\n".join(profile_lines),
        }
    )

    relationship_keywords = [
        "compet",
        "customer",
        "consumer",
        "supplier",
        "supply chain",
        "distribution",
        "distributor",
        "retailer",
        "wholesaler",
        "bottling partner",
        "concentration",
        "raw material",
        "pricing",
    ]

    for index, filing in enumerate(company_profile.get("recent_10k_sections", []), start=1):
        if filing.get("item_1_excerpt"):
            sources.append(
                {
                    "label": f"sec_10k_{index}_item_1_overview",
                    "source_type": "sec_10k_item_1",
                    "text": (
                        f"Filing Date: {filing['filing_date']}\n"
                        f"Filing URL: {filing['filing_url']}\n"
                        f"Item 1 Business Excerpt:\n{filing['item_1_excerpt']}"
                    ),
                }
            )

        item_1_relationship_text = _build_keyword_paragraph_snippet(
            filing.get("item_1_text"),
            keywords=relationship_keywords,
        )
        if item_1_relationship_text:
            sources.append(
                {
                    "label": f"sec_10k_{index}_item_1_relationships",
                    "source_type": "sec_10k_item_1",
                    "text": (
                        f"Filing Date: {filing['filing_date']}\n"
                        f"Item 1 Business Relationship Evidence:\n{item_1_relationship_text}"
                    ),
                }
            )

        item_1a_relationship_text = _build_keyword_paragraph_snippet(
            filing.get("item_1a_text"),
            keywords=relationship_keywords,
        )
        if item_1a_relationship_text:
            sources.append(
                {
                    "label": f"sec_10k_{index}_item_1a_risks",
                    "source_type": "sec_10k_item_1a",
                    "text": (
                        f"Filing Date: {filing['filing_date']}\n"
                        f"Item 1A Risk Factors Relationship Evidence:\n{item_1a_relationship_text}"
                    ),
                }
            )

    return sources


def _extract_relationship_graph_with_llm(ticker, company_profile, source_snippets):
    if not source_snippets:
        return _empty_relationship_graph(ticker, company_profile)

    env()
    combined_sources = []
    for snippet in source_snippets:
        snippet_text = snippet["text"][:2400]
        combined_sources.append(
            f"[{snippet['label']}]\n{snippet_text}"
        )

    extraction_prompt = f"""
        You are extracting a compact relationship graph for {company_profile['company_name']} ({ticker}).

        Use only the SEC-derived source snippets below.
        Only extract relationships that are explicit or strongly supported in the filing text.
        Do not use outside knowledge.

        Allowed relationship types:
        - competitor
        - customer
        - supplier

        If the filing names a clear counterparty group rather than a specific company, you may still create a node.
        Examples: bottling partners, distributors, wholesalers, retailers, contract manufacturers.

        Return JSON only with this exact shape:
        {{
          "nodes": [
            {{
              "name": "entity name",
              "entity_type": "organization_or_group",
              "roles": ["customer"],
              "profile": "short profile",
              "aliases": ["optional alias"],
              "source_labels": ["company_profile"]
            }}
          ],
          "edges": [
            {{
              "target_name": "entity name",
              "relationship_type": "customer",
              "summary": "short relation summary",
              "quantitative_detail": "revenue share or other numeric detail if present",
              "evidence": "short supporting snippet",
              "source_labels": ["sec_10k_1_item_1_relationships"],
              "confidence": 0.0
            }}
          ]
        }}

        If no competitors, customers, or suppliers are explicitly supported, return empty arrays.

        Source snippets:
        {chr(10).join(combined_sources)}
    """

    response = Settings.llm.complete(extraction_prompt)
    extracted_payload = _extract_json_payload(str(response))
    graph = _empty_relationship_graph(ticker, company_profile)
    graph["sources"] = [
        {"label": snippet["label"], "source_type": snippet["source_type"]}
        for snippet in source_snippets
    ]

    node_index = {
        graph["nodes"][0]["id"]: graph["nodes"][0]
    }
    node_lookup = {
        _normalize_entity_key(graph["nodes"][0]["name"]): graph["nodes"][0]["id"]
    }
    edge_lookup = {}

    for raw_node in extracted_payload.get("nodes", []):
        node_name = _clean_profile_field(raw_node.get("name"))
        if not node_name:
            continue

        normalized_key = _normalize_entity_key(node_name)
        node_id = node_lookup.get(normalized_key)
        if not node_id:
            node_id = f"entity::{normalized_key or len(node_lookup)}"
            merged_node = {
                "id": node_id,
                "name": node_name,
                "entity_type": _clean_profile_field(raw_node.get("entity_type")) or "organization",
                "roles": sorted({
                    role
                    for role in (raw_node.get("roles") or [])
                    if _clean_profile_field(role)
                }),
                "profile": _clean_profile_field(raw_node.get("profile")) or "",
                "aliases": sorted({
                    alias
                    for alias in (raw_node.get("aliases") or [])
                    if _clean_profile_field(alias)
                }),
                "source_labels": sorted({
                    label
                    for label in (raw_node.get("source_labels") or [])
                    if _clean_profile_field(label)
                }),
            }
            graph["nodes"].append(merged_node)
            node_index[node_id] = merged_node
            node_lookup[normalized_key] = node_id
        else:
            merged_node = node_index[node_id]
            merged_node["roles"] = sorted(set(merged_node.get("roles", [])) | {
                role for role in (raw_node.get("roles") or []) if _clean_profile_field(role)
            })
            merged_node["aliases"] = sorted(set(merged_node.get("aliases", [])) | {
                alias for alias in (raw_node.get("aliases") or []) if _clean_profile_field(alias)
            })
            merged_node["source_labels"] = sorted(set(merged_node.get("source_labels", [])) | {
                label for label in (raw_node.get("source_labels") or []) if _clean_profile_field(label)
            })
            if not merged_node.get("profile"):
                merged_node["profile"] = _clean_profile_field(raw_node.get("profile")) or ""

    root_node_id = graph["nodes"][0]["id"]
    for raw_edge in extracted_payload.get("edges", []):
        target_name = _clean_profile_field(raw_edge.get("target_name"))
        relationship_type = _clean_profile_field(raw_edge.get("relationship_type"))
        if not target_name or relationship_type not in {"competitor", "customer", "supplier"}:
            continue

        normalized_key = _normalize_entity_key(target_name)
        target_node_id = node_lookup.get(normalized_key)
        if not target_node_id:
            target_node_id = f"entity::{normalized_key or len(node_lookup)}"
            merged_node = {
                "id": target_node_id,
                "name": target_name,
                "entity_type": "organization",
                "roles": [relationship_type],
                "profile": "",
                "aliases": [],
                "source_labels": sorted({
                    label
                    for label in (raw_edge.get("source_labels") or [])
                    if _clean_profile_field(label)
                }),
            }
            graph["nodes"].append(merged_node)
            node_index[target_node_id] = merged_node
            node_lookup[normalized_key] = target_node_id
        else:
            node_index[target_node_id]["roles"] = sorted(set(node_index[target_node_id].get("roles", [])) | {relationship_type})

        edge_key = (root_node_id, target_node_id, relationship_type)
        if edge_key not in edge_lookup:
            merged_edge = {
                "source_node_id": root_node_id,
                "target_node_id": target_node_id,
                "relationship_type": relationship_type,
                "summary": _clean_profile_field(raw_edge.get("summary")) or "",
                "quantitative_detail": _clean_profile_field(raw_edge.get("quantitative_detail")) or "",
                "evidence": _clean_profile_field(raw_edge.get("evidence")) or "",
                "source_labels": sorted({
                    label
                    for label in (raw_edge.get("source_labels") or [])
                    if _clean_profile_field(label)
                }),
                "confidence": float(raw_edge.get("confidence") or 0.0),
            }
            graph["edges"].append(merged_edge)
            edge_lookup[edge_key] = merged_edge
        else:
            merged_edge = edge_lookup[edge_key]
            merged_edge["source_labels"] = sorted(set(merged_edge.get("source_labels", [])) | {
                label
                for label in (raw_edge.get("source_labels") or [])
                if _clean_profile_field(label)
            })
            merged_edge["confidence"] = max(merged_edge.get("confidence", 0.0), float(raw_edge.get("confidence") or 0.0))
            if not merged_edge.get("summary"):
                merged_edge["summary"] = _clean_profile_field(raw_edge.get("summary")) or ""
            if not merged_edge.get("quantitative_detail"):
                merged_edge["quantitative_detail"] = _clean_profile_field(raw_edge.get("quantitative_detail")) or ""
            if not merged_edge.get("evidence"):
                merged_edge["evidence"] = _clean_profile_field(raw_edge.get("evidence")) or ""

    root_node_id = graph["nodes"][0]["id"]
    for node in graph["nodes"]:
        if node["id"] == root_node_id:
            continue

        for relationship_type in {"competitor", "customer", "supplier"}:
            if relationship_type not in set(node.get("roles", [])):
                continue

            edge_key = (root_node_id, node["id"], relationship_type)
            if edge_key in edge_lookup:
                continue

            merged_edge = {
                "source_node_id": root_node_id,
                "target_node_id": node["id"],
                "relationship_type": relationship_type,
                "summary": node.get("profile") or f"Explicitly extracted as a {relationship_type} relationship from SEC filing text.",
                "quantitative_detail": "",
                "evidence": "",
                "source_labels": sorted(set(node.get("source_labels", []))),
                "confidence": 0.35,
            }
            graph["edges"].append(merged_edge)
            edge_lookup[edge_key] = merged_edge

    return graph


def build_company_relationship_graph(ticker, company_profile=None):
    company_profile = company_profile or get_company_profile(ticker)
    try:
        source_snippets = _build_relationship_source_snippets(
            ticker,
            company_profile,
        )
        graph = _extract_relationship_graph_with_llm(ticker, company_profile, source_snippets)
        return _apply_sec_relationship_heuristics(graph, company_profile)
    except Exception as exc:
        print(f"Skipping relationship graph extraction for {ticker}: {exc}")
        graph = _empty_relationship_graph(ticker, company_profile)
        graph["sources"] = [{"label": "company_profile", "source_type": "sec_company_profile"}]
        return _apply_sec_relationship_heuristics(graph, company_profile)


def _relationship_metadata_summary(relationship_graph):
    ticker = (relationship_graph.get("ticker") or "").upper()
    root_node_id = f"company::{ticker}"
    node_map = {
        node.get("id"): node
        for node in relationship_graph.get("nodes", [])
        if node.get("id")
    }

    relation_names = {"competitor": [], "customer": [], "supplier": []}
    concentration_details = {"customer": [], "supplier": [], "competitor": []}

    for node in relationship_graph.get("nodes", []):
        if node.get("id") == root_node_id:
            continue

        counterparty_name = (node.get("name") or "").strip()
        if not counterparty_name:
            continue

        for role in node.get("roles", []):
            if role in relation_names and counterparty_name not in relation_names[role]:
                relation_names[role].append(counterparty_name)

    for edge in relationship_graph.get("edges", []):
        if edge.get("source_node_id") != root_node_id:
            continue

        relationship_type = edge.get("relationship_type")
        if relationship_type not in relation_names:
            continue

        target_node = node_map.get(edge.get("target_node_id"), {})
        counterparty_name = (target_node.get("name") or "").strip()
        if counterparty_name and counterparty_name not in relation_names[relationship_type]:
            relation_names[relationship_type].append(counterparty_name)

        quantitative_detail = (edge.get("quantitative_detail") or "").strip()
        if counterparty_name and quantitative_detail:
            detail = f"{counterparty_name}: {quantitative_detail}"
            if detail not in concentration_details[relationship_type]:
                concentration_details[relationship_type].append(detail)

    def _prune_relation_names(values):
        cleaned_values = []
        for value in sorted(set(values), key=lambda item: (-len(item), item.lower())):
            lowered_value = value.lower()
            if any(lowered_value in existing.lower() for existing in cleaned_values):
                continue
            cleaned_values.append(value)
        return cleaned_values

    return {
        "competitors": _prune_relation_names(relation_names["competitor"]),
        "major_customers": _prune_relation_names(relation_names["customer"]),
        "key_suppliers": _prune_relation_names(relation_names["supplier"]),
        "customer_concentration": concentration_details["customer"],
        "supplier_concentration": concentration_details["supplier"],
        "competitor_share_signals": concentration_details["competitor"],
    }


def _relationship_list_text(values):
    return ", ".join(values) if values else "None explicitly extracted"


def _upsert_relationship(graph, relationship_type, counterparty_name, summary, evidence, source_label, confidence=0.4):
    counterparty_name = _clean_profile_field(counterparty_name)
    if not counterparty_name or relationship_type not in {"competitor", "customer", "supplier"}:
        return

    root_node = graph["nodes"][0]
    root_node_id = root_node["id"]
    normalized_key = _normalize_entity_key(counterparty_name)
    node_lookup = {node["id"]: node for node in graph.get("nodes", [])}
    lookup_by_name = {
        _normalize_entity_key(node.get("name")): node["id"]
        for node in graph.get("nodes", [])
        if node.get("name")
    }

    target_node_id = lookup_by_name.get(normalized_key)
    if not target_node_id:
        target_node_id = f"entity::{normalized_key or len(graph['nodes'])}"
        graph["nodes"].append(
            {
                "id": target_node_id,
                "name": counterparty_name,
                "entity_type": "organization_or_group",
                "roles": [relationship_type],
                "profile": summary or "",
                "aliases": [],
                "source_labels": [source_label],
            }
        )
        node_lookup[target_node_id] = graph["nodes"][-1]
    else:
        node_lookup[target_node_id]["roles"] = sorted(
            set(node_lookup[target_node_id].get("roles", [])) | {relationship_type}
        )
        node_lookup[target_node_id]["source_labels"] = sorted(
            set(node_lookup[target_node_id].get("source_labels", [])) | {source_label}
        )
        if not node_lookup[target_node_id].get("profile"):
            node_lookup[target_node_id]["profile"] = summary or ""

    for edge in graph.get("edges", []):
        if (
            edge.get("source_node_id") == root_node_id
            and edge.get("target_node_id") == target_node_id
            and edge.get("relationship_type") == relationship_type
        ):
            if not edge.get("summary"):
                edge["summary"] = summary or ""
            if not edge.get("evidence"):
                edge["evidence"] = evidence or ""
            edge["source_labels"] = sorted(set(edge.get("source_labels", [])) | {source_label})
            edge["confidence"] = max(edge.get("confidence", 0.0), confidence)
            return

    graph["edges"].append(
        {
            "source_node_id": root_node_id,
            "target_node_id": target_node_id,
            "relationship_type": relationship_type,
            "summary": summary or "",
            "quantitative_detail": "",
            "evidence": evidence or "",
            "source_labels": [source_label],
            "confidence": confidence,
        }
    )


def _extract_sentence_with_pattern(text, pattern):
    if not text:
        return ""
    match = re.search(rf"([^.]*{pattern}[^.]*)\.", text, flags=re.IGNORECASE)
    if match:
        return _clean_section_text(match.group(1))
    return ""


def _apply_sec_relationship_heuristics(graph, company_profile):
    texts = []
    if company_profile.get("description"):
        texts.append(("company_profile", company_profile["description"]))
    for index, filing in enumerate(company_profile.get("recent_10k_sections", []), start=1):
        if filing.get("item_1_text"):
            texts.append((f"sec_10k_{index}_item_1", filing["item_1_text"]))
        if filing.get("item_1a_text"):
            texts.append((f"sec_10k_{index}_item_1a", filing["item_1a_text"]))

    heuristic_patterns = [
        (
            r"independent bottling partners(?:,\s*distributors,\s*wholesalers\s*and\s*retailers)?",
            "customer",
            "Distribution and customer-facing network referenced in SEC filing.",
        ),
        (
            r"key retail or foodservice customers",
            "customer",
            "Key retail or foodservice customers referenced in SEC filing.",
        ),
        (
            r"distributors,\s*wholesalers\s*and\s*retailers",
            "customer",
            "Distribution channel partners referenced in SEC filing.",
        ),
        (
            r"third[- ]party suppliers?",
            "supplier",
            "Third-party suppliers referenced in SEC filing.",
        ),
        (
            r"raw material suppliers?",
            "supplier",
            "Raw material suppliers referenced in SEC filing.",
        ),
        (
            r"contract manufacturers?",
            "supplier",
            "Contract manufacturing partners referenced in SEC filing.",
        ),
    ]

    for source_label, text in texts:
        lowered_text = text.lower()
        for pattern, relationship_type, summary in heuristic_patterns:
            match = re.search(pattern, lowered_text, flags=re.IGNORECASE)
            if not match:
                continue

            matched_value = text[match.start():match.end()]
            evidence = _extract_sentence_with_pattern(text, pattern) or matched_value
            _upsert_relationship(
                graph,
                relationship_type=relationship_type,
                counterparty_name=matched_value,
                summary=summary,
                evidence=evidence,
                source_label=source_label,
                confidence=0.45,
            )

    return graph


def build_relationship_graph_docs(relationship_graph):
    documents = []
    company_name = relationship_graph.get("company_name") or relationship_graph.get("ticker")
    ticker = relationship_graph.get("ticker")
    nodes = relationship_graph.get("nodes", [])
    edges = relationship_graph.get("edges", [])
    node_map = {node["id"]: node for node in nodes}
    relationship_metadata = _relationship_metadata_summary(relationship_graph)

    relation_buckets = {"competitor": [], "customer": [], "supplier": []}
    for edge in edges:
        target_node = node_map.get(edge["target_node_id"], {})
        relation_buckets.setdefault(edge["relationship_type"], []).append(target_node.get("name", "Unknown"))

    overview_lines = [
        f"Company: {company_name}",
        f"Ticker: {ticker}",
        f"Known competitors: {_relationship_list_text(sorted(set(relation_buckets.get('competitor', []))))}",
        f"Known customers: {_relationship_list_text(sorted(set(relation_buckets.get('customer', []))))}",
        f"Known suppliers: {_relationship_list_text(sorted(set(relation_buckets.get('supplier', []))))}",
        f"Customer concentration detail: {_relationship_list_text(relationship_metadata['customer_concentration'])}",
        f"Supplier concentration detail: {_relationship_list_text(relationship_metadata['supplier_concentration'])}",
    ]
    documents.append(
        Document(
            text="**Relationship Graph Overview**\n\n" + "\n".join(overview_lines),
            metadata={
                "ticker": ticker,
                "type": "relationship_graph_overview",
                "company_name": company_name,
                "source": "SEC 10-K Item 1 / Item 1A + LLM extraction",
                "relationship_edge_count": len(edges),
                "relationship_node_count": len(nodes),
                "competitors": relationship_metadata["competitors"],
                "major_customers": relationship_metadata["major_customers"],
                "key_suppliers": relationship_metadata["key_suppliers"],
                "customer_concentration": relationship_metadata["customer_concentration"],
                "supplier_concentration": relationship_metadata["supplier_concentration"],
            },
        )
    )

    for node in nodes:
        if node["id"] == f"company::{ticker}":
            continue

        documents.append(
            Document(
                text=(
                    "**Relationship Graph Node**\n\n"
                    f"Company: {company_name} ({ticker})\n"
                    f"Entity: {node['name']}\n"
                    f"Entity Type: {node.get('entity_type', 'organization')}\n"
                    f"Roles: {', '.join(node.get('roles', [])) or 'Unknown'}\n"
                    f"Profile: {node.get('profile') or 'No short profile extracted.'}\n"
                    f"Source Labels: {', '.join(node.get('source_labels', [])) or 'None'}"
                ),
                metadata={
                    "ticker": ticker,
                    "type": "relationship_graph_node",
                    "entity_name": node["name"],
                    "entity_roles": ", ".join(node.get("roles", [])),
                    "source": "SEC 10-K Item 1 / Item 1A + LLM extraction",
                },
            )
        )

    for edge in edges:
        target_node = node_map.get(edge["target_node_id"], {})
        documents.append(
            Document(
                text=(
                    "**Relationship Graph Edge**\n\n"
                    f"Company: {company_name} ({ticker})\n"
                    f"Relationship: {edge['relationship_type']}\n"
                    f"Counterparty: {target_node.get('name', 'Unknown')}\n"
                    f"Summary: {edge.get('summary') or 'No summary extracted.'}\n"
                    f"Quantitative Detail: {edge.get('quantitative_detail') or 'Not stated'}\n"
                    f"Evidence: {edge.get('evidence') or 'No evidence snippet extracted.'}\n"
                    f"Source Labels: {', '.join(edge.get('source_labels', [])) or 'None'}\n"
                    f"Confidence: {edge.get('confidence', 0.0):.2f}"
                ),
                metadata={
                    "ticker": ticker,
                    "type": "relationship_graph_edge",
                    "relationship_type": edge["relationship_type"],
                    "counterparty_name": target_node.get("name", "Unknown"),
                    "source": "SEC 10-K Item 1 / Item 1A + LLM extraction",
                },
            )
        )

    return documents


def _persist_relationship_graph(relationship_graph, persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    graph_path = os.path.join(persist_dir, RELATIONSHIP_GRAPH_FILENAME)
    with open(graph_path, "w", encoding="utf-8") as graph_file:
        json.dump(relationship_graph, graph_file, ensure_ascii=True, indent=2)
    return graph_path


# Modify your existing function to allow fetching all periods
def get_historical_financial_indicators(ticker, frequency='quarterly', max_periods=None):
    """
    Fetch historical financial indicators from yfinance (quarterly or annual),
    optionally limited to the most recent `max_periods` periods.

    If max_periods=None, fetch all available.
    """
    stock = yf.Ticker(ticker.upper())

    if frequency.lower() == 'quarterly':
        income_src = stock.quarterly_income_stmt
        balance_src = stock.quarterly_balance_sheet
        cashflow_src = stock.quarterly_cashflow
        freq_label = "Quarterly"
    else:
        income_src = stock.income_stmt
        balance_src = stock.balance_sheet
        cashflow_src = stock.cashflow
        freq_label = "Annual"

    # Early exits for missing data
    if income_src is None or income_src.empty:
        print(f"{ticker}: No {freq_label} income statement available")
        return pd.DataFrame()
    if balance_src is None or balance_src.empty:
        print(f"{ticker}: No {freq_label} balance sheet available")
        return pd.DataFrame()
    if cashflow_src is None or cashflow_src.empty:
        print(f"{ticker}: No {freq_label} cash flow statement available")
        return pd.DataFrame()

    # Transpose to have dates as index
    income = income_src.T
    balance = balance_src.T
    cashflow = cashflow_src.T

    # Limit to most recent if specified
    if max_periods is not None:
        income = income.iloc[:max_periods]
        balance = balance.iloc[:max_periods]
        cashflow = cashflow.iloc[:max_periods]

    # Find common dates
    common_dates = income.index.intersection(balance.index).intersection(cashflow.index)

    if len(common_dates) == 0:
        print(f"{ticker} ({freq_label}): No overlapping period dates")
        return pd.DataFrame()

    income = income.loc[common_dates]
    balance = balance.loc[common_dates]
    cashflow = cashflow.loc[common_dates]

    # Build the indicators DataFrame
    df = pd.DataFrame(index=common_dates)

    df['Total Revenue'] = income.get('Total Revenue', pd.NA)
    df['Gross Profit'] = income.get('Gross Profit', pd.NA)
    df['Operating Income'] = income.get('Operating Income', pd.NA)
    df['Net Income'] = income.get('Net Income', pd.NA)
    df['Basic EPS'] = income.get('Basic EPS', pd.NA)

    df['Total Assets'] = balance.get('Total Assets', pd.NA)
    df['Total Liabilities'] = balance.get('Total Liabilities Net Minority Interest', pd.NA)
    df['Shareholders Equity'] = balance.get('Common Stock Equity', pd.NA)

    df['Operating Cash Flow'] = cashflow.get('Operating Cash Flow', pd.NA)
    df['Free Cash Flow'] = cashflow.get('Free Cash Flow', pd.NA)

    # Computed ratios
    df['Gross Margin'] = df['Gross Profit'] / df['Total Revenue']
    df['Operating Margin'] = df['Operating Income'] / df['Total Revenue']
    df['Net Margin'] = df['Net Income'] / df['Total Revenue']
    df['ROE'] = df['Net Income'] / df['Shareholders Equity']
    df['ROA'] = df['Net Income'] / df['Total Assets']
    df['Debt to Equity'] = df['Total Liabilities'] / df['Shareholders Equity']

    # Finalize
    df = df.sort_index(ascending=False)  # most recent first
    df = df.reset_index().rename(columns={'index': 'Period End Date'})
    df['Ticker'] = ticker.upper()
    df['Frequency'] = freq_label

    # Format date as string for consistency
    df['Period End Date'] = pd.to_datetime(df['Period End Date']).dt.strftime('%Y-%m-%d')

    return df

def initiate_sql_table(db_path=DEFAULT_STOCK_DB_PATH):
    # it deletes old data...run with caution
    check = input("it deletes old data in stock_data file...run with caution, are you sure you want to delete and init? Y/N")
    if check.upper() == 'Y':
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop the old table (warning: deletes all existing data!)
        cursor.execute("DROP TABLE IF EXISTS financial_indicators")

        # Now recreate with the correct columns (your current CREATE statement)
        create_table_query = """
        CREATE TABLE financial_indicators (
            Ticker TEXT,
            Frequency TEXT,
            `Period End Date` TEXT,
            `Total Revenue` FLOAT,
            `Gross Profit` FLOAT,
            `Operating Income` FLOAT,
            `Net Income` FLOAT,
            `Basic EPS` FLOAT,
            `Total Assets` FLOAT,
            `Total Liabilities` FLOAT,
            `Shareholders Equity` FLOAT,
            `Operating Cash Flow` FLOAT,
            `Free Cash Flow` FLOAT,
            `Gross Margin` FLOAT,
            `Operating Margin` FLOAT,
            `Net Margin` FLOAT,
            ROE FLOAT,
            ROA FLOAT,
            `Debt to Equity` FLOAT
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()
        print("Table dropped and recreated successfully.")
    else:
        print('sql table not initialized...')
# Example integration into build_documents (modify as per your existing setup)
def build_financial_docs(
    ticker,
    db_path=DEFAULT_STOCK_DB_PATH,
    max_quarters=12,
    max_annual=10,
    filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR):
    documents = []
    conn = sqlite3.connect(db_path)

    ticker = ticker.upper()
    sec_archive = _sync_sec_filing_archive(
        ticker,
        filings_base_dir=filings_base_dir,
        max_10k_filings=10,
        max_10q_filings=12,
    )
    company_profile = _build_company_profile_from_sec_archive(ticker, sec_archive)

    profile_lines = [
        f"Company: {company_profile['company_name']}",
        f"Ticker: {ticker}",
        f"Sector: {company_profile['sector'] or 'Unknown'}",
        f"Industry: {company_profile['industry'] or 'Unknown'}",
        f"Source: {company_profile['source']}",
        f"Stored 10-K filings: {len(company_profile.get('ten_k_filings', []))}",
        f"Stored 10-Q filings: {len(company_profile.get('ten_q_filings', []))}",
    ]
    if company_profile.get("latest_filing_date"):
        profile_lines.append(f"Latest 10-K Filing Date: {company_profile['latest_filing_date']}")
    if company_profile.get("latest_filing_url"):
        profile_lines.append(f"Latest 10-K Filing URL: {company_profile['latest_filing_url']}")
    if company_profile["description"]:
        profile_lines.append("")
        profile_lines.append("Business Description (Recent SEC Item 1 Excerpt):")
        profile_lines.append(company_profile["description"])

    documents.append(
        Document(
            text="**Company Overview (SEC 10-K Item 1 Business)**\n\n" + "\n".join(profile_lines),
            metadata={
                "ticker": ticker,
                "type": "company_profile",
                "company_name": company_profile["company_name"],
                "sector": company_profile["sector"] or "Unknown",
                "industry": company_profile["industry"] or "Unknown",
                "source": company_profile["source"],
                "filing_date": company_profile.get("latest_filing_date"),
                "filing_url": company_profile.get("latest_filing_url"),
                "ten_k_count": len(company_profile.get("ten_k_filings", [])),
                "ten_q_count": len(company_profile.get("ten_q_filings", [])),
            }
        )
    )

    for filing_record in company_profile.get("ten_k_filings", []):
        documents.extend(_documents_from_payloads(filing_record.get("documents", [])))

    for filing_record in company_profile.get("ten_q_filings", []):
        documents.extend(_documents_from_payloads(filing_record.get("documents", [])))

    for freq, max_keep in [('Quarterly', max_quarters), ('Annual', max_annual)]:
        query = """
        SELECT * FROM financial_indicators
        WHERE Ticker = ? AND Frequency = ?
        ORDER BY `Period End Date` DESC
        LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker, freq, max_keep))

        if df.empty:
            continue

        md_table = df.to_markdown(index=False)

        text = (
            f"**Latest {len(df)} {freq} Financial Indicators - {ticker}**\n\n"
            f"Company: {company_profile['company_name']}\n"
            f"Sector: {company_profile['sector'] or 'Unknown'}\n"
            f"Industry: {company_profile['industry'] or 'Unknown'}\n\n"
            f"Most recent: {df['Period End Date'].iloc[0]}\n"
            f"Oldest shown: {df['Period End Date'].iloc[-1]}\n\n"
            f"{md_table}\n\n"
            f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
        )

        doc = Document(
            text=text,
            metadata={
                "ticker": ticker,
                "type": "financial_indicators",
                "company_name": company_profile["company_name"],
                "sector": company_profile["sector"] or "Unknown",
                "industry": company_profile["industry"] or "Unknown",
                "frequency": freq,
                "periods": len(df),
                "most_recent": df['Period End Date'].iloc[0],
                "source": "yfinance + SQLite",
                "table_rows": len(df),
                "ten_k_count": len(company_profile.get("ten_k_filings", [])),
                "ten_q_count": len(company_profile.get("ten_q_filings", [])),
            }
        )
        documents.append(doc)

    conn.close()
    return documents

def update_financial_records(ticker, db_path=DEFAULT_STOCK_DB_PATH):
    """
    Checks and updates financial records for a ticker in SQLite storage.

    - Fetches latest from yfinance (all available periods).
    - For annual: keep most recent 10 years.
    - For quarterly: keep most recent 12 quarters.
    - If up to date (latest period matches), do nothing.
    - Else: add new periods, then trim to max keep (deque-like: remove oldest if excess).
    """
    conn = sqlite3.connect(db_path)
    ticker = ticker.upper()

    for freq, max_keep in [('annual', 10), ('quarterly', 12)]:
        freq_label = freq.capitalize()

        # Fetch all available latest data from yfinance
        df_new = get_historical_financial_indicators(ticker, frequency=freq, max_periods=None)

        if df_new.empty:
            print(f"Skipping update for {ticker} {freq_label}: No new data available")
            continue

        # Read existing from DB
        query = """
        SELECT * FROM financial_indicators
        WHERE Ticker = ? AND Frequency = ?
        ORDER BY `Period End Date` DESC
        """
        df_existing = pd.read_sql_query(query, conn, params=(ticker, freq_label))

        if not df_existing.empty:
            latest_existing_date = df_existing['Period End Date'].iloc[0]
            latest_new_date = df_new['Period End Date'].iloc[0]

            if latest_existing_date >= latest_new_date:
                print(f"{ticker} {freq_label} is up to date (latest: {latest_existing_date})")
                continue  # No update needed

        # Combine existing + new, drop duplicates based on date
        if df_existing.empty:
            df_combined = df_new.copy()
        elif df_new.empty:
            df_combined = df_existing.copy()
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)        # keep newest if conflict

        # Sort by date DESC (most recent first)
        df_combined['Period End Date'] = pd.to_datetime(df_combined['Period End Date'])
        df_combined = df_combined.sort_values('Period End Date', ascending=False).reset_index(drop=True)

        # Trim to max_keep (remove oldest if more)
        df_trimmed = df_combined.head(max_keep)

        # Overwrite in DB: delete old, insert new
        delete_query = """
        DELETE FROM financial_indicators
        WHERE Ticker = ? AND Frequency = ?
        """
        conn.execute(delete_query, (ticker, freq_label))

        df_trimmed.to_sql('financial_indicators', conn, if_exists='append', index=False)

        print(f"Updated {ticker} {freq_label}: Kept {len(df_trimmed)} periods (latest: {df_trimmed['Period End Date'].iloc[0].strftime('%Y-%m-%d')})")

    conn.commit()
    conn.close()

def refresh_ticker_data_and_index(
    ticker: str,
    db_path=DEFAULT_STOCK_DB_PATH,
    max_quarters=12,
    max_annual=10,
    filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR,
    storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR):
    """
    End-to-end refresh for one ticker:
    1. Update structured data in SQLite
    2. Build fresh LlamaIndex Documents from the DB
    3. Upsert into the vector index
    """
    ticker = ticker.upper()

    # Step 1: refresh numbers in SQL
    update_financial_records(ticker, db_path=db_path)

    os.makedirs(storage_base_dir, exist_ok=True)
    persist_dir = os.path.join(storage_base_dir, ticker)

    # Step 2: build documents from the fresh DB
    docs = build_financial_docs(
        ticker,
        db_path=db_path,
        max_quarters=max_quarters,
        max_annual=max_annual,
        filings_base_dir=filings_base_dir,
    )

    if not docs:
        print(f"No documents generated for {ticker} — skipping index update")
        return None

    # Step 3: rebuild vector store so old Yahoo-based docs do not linger
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)
    _reset_persist_dir(persist_dir)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)

    print(f"Index successfully refreshed for {ticker}")


#refresh_ticker_data_and_index('aapl')
