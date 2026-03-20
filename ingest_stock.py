import json
import os
import re
import sqlite3
import yfinance as yf
import requests
from llama_index.core import (Settings, Document, VectorStoreIndex, StorageContext,load_index_from_storage)
from datetime import datetime
from html import unescape
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from dotenv import load_dotenv


load_dotenv("config.env")

DEFAULT_STOCK_DB_PATH = os.getenv("STOCK_SQL_DB_PATH", "stock_data.db")
DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")
RELATIONSHIP_GRAPH_FILENAME = "relationship_graph.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dashes}/{primary_document}"
DEFAULT_SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "RAGResearch contact@example.com")
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
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate",
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
    return response.text


def _load_sec_ticker_map():
    global _SEC_TICKER_MAP
    if _SEC_TICKER_MAP is not None:
        return _SEC_TICKER_MAP

    ticker_map = {}
    for item in _sec_get_json(SEC_TICKERS_URL).values():
        ticker = _clean_profile_field(item.get("ticker"))
        if not ticker:
            continue
        ticker_map[ticker.upper()] = item

    _SEC_TICKER_MAP = ticker_map
    return _SEC_TICKER_MAP


def _truncate_text(text, max_chars=2400):
    if not text:
        return None
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rsplit(" ", 1)[0] + "..."


def _derive_sector_from_sic(sic_code, sic_description):
    sic_description = (sic_description or "").lower()
    sic_code = int(sic_code or 0)

    keyword_map = [
        ("Consumer Defensive", ["beverage", "food", "tobacco", "household", "personal care", "consumer goods"]),
        ("Consumer Cyclical", ["auto", "retail", "restaurant", "apparel", "footwear", "hotel", "leisure", "consumer products"]),
        ("Technology", ["software", "semiconductor", "computer", "communications equipment", "electronic", "internet", "technology"]),
        ("Healthcare", ["pharmaceutical", "biotechnology", "medical", "health", "diagnostic"]),
        ("Financial Services", ["bank", "insurance", "capital", "asset management", "financial"]),
        ("Real Estate", ["real estate", "reit"]),
        ("Communication Services", ["media", "telecom", "broadcasting", "entertainment"]),
        ("Utilities", ["utility", "electric services", "gas services", "water supply"]),
        ("Energy", ["oil", "gas", "petroleum", "pipeline", "drilling"]),
        ("Basic Materials", ["chemical", "steel", "metal", "mining", "paper", "forest"]),
        ("Industrials", ["machinery", "transportation", "aerospace", "defense", "industrial", "manufacturing"]),
    ]
    for sector_name, keywords in keyword_map:
        if any(keyword in sic_description for keyword in keywords):
            return sector_name

    if 100 <= sic_code <= 999:
        return "Basic Materials"
    if 1000 <= sic_code <= 1499:
        return "Energy"
    if 1500 <= sic_code <= 1799:
        return "Industrials"
    if 2000 <= sic_code <= 3999:
        return "Industrials"
    if 4000 <= sic_code <= 4999:
        return "Industrials"
    if 5000 <= sic_code <= 5999:
        return "Consumer Cyclical"
    if 6000 <= sic_code <= 6799:
        return "Financial Services"
    if 7000 <= sic_code <= 8999:
        return "Services"
    if 9100 <= sic_code <= 9999:
        return "Government"

    return None


def _html_to_text_preserve_lines(html):
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)</(?:p|div|tr|li|table|section|article|h1|h2|h3|h4|h5|h6)>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _clean_section_text(text):
    if not text:
        return None
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _find_best_section(text, start_patterns, end_patterns, min_length=2000):
    upper_text = text.upper()
    start_positions = []
    end_positions = []

    for pattern in start_patterns:
        start_positions.extend(match.start() for match in re.finditer(pattern, upper_text))
    for pattern in end_patterns:
        end_positions.extend(match.start() for match in re.finditer(pattern, upper_text))

    start_positions = sorted(set(start_positions))
    end_positions = sorted(set(end_positions))
    if not start_positions or not end_positions:
        return None

    fallback_section = None
    fallback_length = -1

    for start_position in start_positions:
        end_position = next((value for value in end_positions if value > start_position + 50), None)
        if end_position is None:
            continue

        section_text = _clean_section_text(text[start_position:end_position])
        if not section_text:
            continue

        preview = section_text[:350].upper()
        if any(re.search(pattern, preview[20:]) for pattern in end_patterns) and len(section_text) < min_length:
            continue

        if len(section_text) > fallback_length:
            fallback_section = section_text
            fallback_length = len(section_text)

        if len(section_text) >= min_length:
            return section_text

    return fallback_section


def _split_paragraphs(text):
    if not text:
        return []
    paragraphs = []
    for chunk in re.split(r"\n{2,}", text):
        cleaned_chunk = re.sub(r"\s+", " ", chunk).strip()
        if len(cleaned_chunk) >= 80:
            paragraphs.append(cleaned_chunk)
    return paragraphs


def _build_keyword_paragraph_snippet(text, keywords, max_paragraphs=6):
    paragraphs = _split_paragraphs(text)
    matches = []

    for paragraph in paragraphs:
        lowered_paragraph = paragraph.lower()
        score = sum(keyword in lowered_paragraph for keyword in keywords)
        if score:
            matches.append((score, paragraph))

    matches.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected = []
    seen = set()
    for _, paragraph in matches:
        if paragraph in seen:
            continue
        seen.add(paragraph)
        selected.append(paragraph)
        if len(selected) >= max_paragraphs:
            break

    return "\n\n".join(selected)


def _build_recent_10k_sections(ticker, max_filings=2):
    ticker = ticker.upper()
    ticker_map = _load_sec_ticker_map()
    ticker_info = ticker_map.get(ticker)
    if not ticker_info:
        raise ValueError(f"Ticker {ticker} was not found in SEC company_tickers.json")

    cik = str(ticker_info["cik_str"]).zfill(10)
    submissions = _sec_get_json(SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
    recent_filings = submissions.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])

    filing_sections = []
    for index, form in enumerate(forms):
        if form != "10-K":
            continue

        accession_number = recent_filings["accessionNumber"][index]
        primary_document = recent_filings["primaryDocument"][index]
        filing_date = recent_filings["filingDate"][index]
        accession_no_dashes = accession_number.replace("-", "")
        filing_url = SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE.format(
            cik_no_zero=int(cik),
            accession_no_dashes=accession_no_dashes,
            primary_document=primary_document,
        )

        filing_html = _sec_get_text(filing_url)
        filing_text = _html_to_text_preserve_lines(filing_html)
        item_1_text = _find_best_section(
            filing_text,
            start_patterns=[r"ITEM\s+1\.?\s+BUSINESS"],
            end_patterns=[r"ITEM\s+1A\.?\s+RISK\s+FACTORS", r"ITEM\s+2\.?\s+PROPERTIES"],
            min_length=2500,
        )
        item_1a_text = _find_best_section(
            filing_text,
            start_patterns=[r"ITEM\s+1A\.?\s+RISK\s+FACTORS"],
            end_patterns=[r"ITEM\s+1B\.?\s+UNRESOLVED\s+STAFF\s+COMMENTS", r"ITEM\s+1C\.?\s+CYBERSECURITY", r"ITEM\s+2\.?\s+PROPERTIES"],
            min_length=1800,
        )

        filing_sections.append(
            {
                "ticker": ticker,
                "company_name": _clean_profile_field(submissions.get("name")) or ticker_info.get("title") or ticker,
                "cik": cik,
                "sic": _clean_profile_field(submissions.get("sic")),
                "sic_description": _clean_profile_field(submissions.get("sicDescription")),
                "filing_date": filing_date,
                "form": form,
                "accession_number": accession_number,
                "filing_url": filing_url,
                "item_1_text": item_1_text,
                "item_1_excerpt": _truncate_text(item_1_text, max_chars=3200),
                "item_1a_text": item_1a_text,
                "item_1a_excerpt": _truncate_text(item_1a_text, max_chars=2500),
            }
        )

        if len(filing_sections) >= max_filings:
            break

    if not filing_sections:
        raise ValueError(f"No recent 10-K filings were found for {ticker}")

    return filing_sections


def _get_yahoo_taxonomy_fallback(ticker):
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)
    try:
        info = stock.get_info()
    except Exception:
        return {}

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

    yahoo_fallback = _get_yahoo_taxonomy_fallback(ticker)
    latest_filing = recent_10k_sections[0] if recent_10k_sections else {}

    company_name = (
        latest_filing.get("company_name")
        or yahoo_fallback.get("company_name")
        or ticker
    )
    industry = (
        latest_filing.get("sic_description")
        or yahoo_fallback.get("industry")
    )
    sector = (
        _derive_sector_from_sic(latest_filing.get("sic"), latest_filing.get("sic_description"))
        or yahoo_fallback.get("sector")
    )
    description = latest_filing.get("item_1_excerpt")

    if latest_filing:
        source = "SEC EDGAR 10-K Item 1 Business"
    else:
        source = "Yahoo Finance via yfinance"

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
    ticker = ticker.upper()
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

        item_1_evidence = _build_keyword_paragraph_snippet(
            filing.get("item_1_text"),
            relationship_keywords,
            max_paragraphs=8,
        )
        if item_1_evidence:
            sources.append(
                {
                    "label": f"sec_10k_{index}_item_1_relationships",
                    "source_type": "sec_10k_item_1",
                    "text": (
                        f"Filing Date: {filing['filing_date']}\n"
                        "Item 1 Business Relationship Evidence:\n"
                        f"{item_1_evidence}"
                    ),
                }
            )

        item_1a_evidence = _build_keyword_paragraph_snippet(
            filing.get("item_1a_text"),
            relationship_keywords,
            max_paragraphs=8,
        )
        if item_1a_evidence:
            sources.append(
                {
                    "label": f"sec_10k_{index}_item_1a_risks",
                    "source_type": "sec_10k_item_1a",
                    "text": (
                        f"Filing Date: {filing['filing_date']}\n"
                        "Item 1A Risk Factors Relationship Evidence:\n"
                        f"{item_1a_evidence}"
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
        snippet_text = snippet["text"][:1600]
        combined_sources.append(
            f"[{snippet['label']}]\n{snippet_text}"
        )

    extraction_prompt = f"""
You are extracting a compact relationship graph for {company_profile['company_name']} ({ticker}).

Use only the source snippets below.
Only extract relationships that are explicit or strongly supported in the text.
Do not use outside knowledge.

Allowed relationship types:
- competitor
- customer
- supplier

Return JSON only with this exact shape:
{{
  "nodes": [
    {{
      "name": "entity name or clearly defined counterparty group",
      "entity_type": "organization_or_group",
      "roles": ["competitor"],
      "profile": "short profile",
      "aliases": ["optional alias"],
      "source_labels": ["company_profile"]
    }}
  ],
  "edges": [
    {{
      "target_name": "entity name",
      "relationship_type": "competitor",
      "summary": "short relation summary",
      "quantitative_detail": "revenue share or other numeric detail if present",
      "evidence": "short supporting snippet",
      "source_labels": ["news_1"],
      "confidence": 0.0
    }}
  ]
}}

If no competitors, customers, or suppliers are explicitly supported, return empty arrays.
If the filing names a counterparty group like bottling partners, distributors, retailers, wholesalers, or suppliers without naming a specific company, you may still create a node for that group.

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
                "entity_type": "organization_or_group",
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

    for node in graph["nodes"]:
        if node["id"] == root_node_id:
            continue

        for relationship_type in node.get("roles", []):
            if relationship_type not in {"competitor", "customer", "supplier"}:
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
        return _extract_relationship_graph_with_llm(ticker, company_profile, source_snippets)
    except Exception as exc:
        print(f"Skipping relationship graph extraction for {ticker}: {exc}")
        graph = _empty_relationship_graph(ticker, company_profile)
        graph["sources"] = [{"label": "company_profile", "source_type": "sec_company_profile"}]
        return graph


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

    return {
        "competitors": relation_names["competitor"],
        "major_customers": relation_names["customer"],
        "key_suppliers": relation_names["supplier"],
        "customer_concentration": concentration_details["customer"],
        "supplier_concentration": concentration_details["supplier"],
        "competitor_share_signals": concentration_details["competitor"],
    }


def _relationship_list_text(values):
    return ", ".join(values) if values else "None explicitly extracted"


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
    max_annual=8,
    include_relationship_graph=True,
    return_relationship_graph=False):
    documents = []
    conn = sqlite3.connect(db_path)

    ticker = ticker.upper()
    company_profile = get_company_profile(ticker)
    relationship_graph = None
    relationship_metadata = {
        "competitors": [],
        "major_customers": [],
        "key_suppliers": [],
        "customer_concentration": [],
        "supplier_concentration": [],
        "competitor_share_signals": [],
    }

    if include_relationship_graph:
        relationship_graph = build_company_relationship_graph(
            ticker,
            company_profile=company_profile,
        )
        relationship_metadata = _relationship_metadata_summary(relationship_graph)

    profile_lines = [
        f"Company: {company_profile['company_name']}",
        f"Ticker: {ticker}",
        f"Sector: {company_profile['sector'] or 'Unknown'}",
        f"Industry: {company_profile['industry'] or 'Unknown'}",
        f"Source: {company_profile['source']}",
        f"Competitors: {_relationship_list_text(relationship_metadata['competitors'])}",
        f"Major Customers: {_relationship_list_text(relationship_metadata['major_customers'])}",
        f"Key Suppliers: {_relationship_list_text(relationship_metadata['key_suppliers'])}",
    ]
    if relationship_metadata["customer_concentration"]:
        profile_lines.append(
            f"Customer Concentration Detail: {_relationship_list_text(relationship_metadata['customer_concentration'])}"
        )
    if relationship_metadata["supplier_concentration"]:
        profile_lines.append(
            f"Supplier Concentration Detail: {_relationship_list_text(relationship_metadata['supplier_concentration'])}"
        )
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
                "competitors": relationship_metadata["competitors"],
                "major_customers": relationship_metadata["major_customers"],
                "key_suppliers": relationship_metadata["key_suppliers"],
                "customer_concentration": relationship_metadata["customer_concentration"],
                "supplier_concentration": relationship_metadata["supplier_concentration"],
            }
        )
    )

    for filing in company_profile.get("recent_10k_sections", []):
        if filing.get("item_1_text"):
            documents.append(
                Document(
                    text=(
                        "**SEC 10-K Item 1 Business**\n\n"
                        f"Company: {company_profile['company_name']}\n"
                        f"Ticker: {ticker}\n"
                        f"Filing Date: {filing['filing_date']}\n"
                        f"Filing URL: {filing['filing_url']}\n"
                        f"Sector: {company_profile['sector'] or 'Unknown'}\n"
                        f"Industry: {company_profile['industry'] or 'Unknown'}\n\n"
                        f"{filing['item_1_text']}"
                    ),
                    metadata={
                        "ticker": ticker,
                        "type": "sec_10k_item_1",
                        "company_name": company_profile["company_name"],
                        "sector": company_profile["sector"] or "Unknown",
                        "industry": company_profile["industry"] or "Unknown",
                        "filing_date": filing["filing_date"],
                        "filing_url": filing["filing_url"],
                        "source": "SEC 10-K Item 1 Business",
                        "competitors": relationship_metadata["competitors"],
                        "major_customers": relationship_metadata["major_customers"],
                        "key_suppliers": relationship_metadata["key_suppliers"],
                        "customer_concentration": relationship_metadata["customer_concentration"],
                        "supplier_concentration": relationship_metadata["supplier_concentration"],
                    }
                )
            )

        if filing.get("item_1a_text"):
            documents.append(
                Document(
                    text=(
                        "**SEC 10-K Item 1A Risk Factors**\n\n"
                        f"Company: {company_profile['company_name']}\n"
                        f"Ticker: {ticker}\n"
                        f"Filing Date: {filing['filing_date']}\n"
                        f"Filing URL: {filing['filing_url']}\n"
                        f"Sector: {company_profile['sector'] or 'Unknown'}\n"
                        f"Industry: {company_profile['industry'] or 'Unknown'}\n\n"
                        f"{filing['item_1a_text']}"
                    ),
                    metadata={
                        "ticker": ticker,
                        "type": "sec_10k_item_1a",
                        "company_name": company_profile["company_name"],
                        "sector": company_profile["sector"] or "Unknown",
                        "industry": company_profile["industry"] or "Unknown",
                        "filing_date": filing["filing_date"],
                        "filing_url": filing["filing_url"],
                        "source": "SEC 10-K Item 1A Risk Factors",
                        "competitors": relationship_metadata["competitors"],
                        "major_customers": relationship_metadata["major_customers"],
                        "key_suppliers": relationship_metadata["key_suppliers"],
                        "customer_concentration": relationship_metadata["customer_concentration"],
                        "supplier_concentration": relationship_metadata["supplier_concentration"],
                    }
                )
            )

    if include_relationship_graph:
        documents.extend(build_relationship_graph_docs(relationship_graph))

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
                "competitors": relationship_metadata["competitors"],
                "major_customers": relationship_metadata["major_customers"],
                "key_suppliers": relationship_metadata["key_suppliers"],
                "customer_concentration": relationship_metadata["customer_concentration"],
                "supplier_concentration": relationship_metadata["supplier_concentration"],
            }
        )
        documents.append(doc)

    conn.close()
    if return_relationship_graph:
        return documents, relationship_graph
    return documents

def update_financial_records(ticker, db_path=DEFAULT_STOCK_DB_PATH):
    """
    Checks and updates financial records for a ticker in SQLite storage.

    - Fetches latest from yfinance (all available periods).
    - For annual: keep most recent 8 years.
    - For quarterly: keep most recent 12 quarters.
    - If up to date (latest period matches), do nothing.
    - Else: add new periods, then trim to max keep (deque-like: remove oldest if excess).
    """
    conn = sqlite3.connect(db_path)
    ticker = ticker.upper()

    for freq, max_keep in [('annual', 8), ('quarterly', 12)]:
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
    max_annual=8,
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
    docs, relationship_graph = build_financial_docs(
        ticker,
        db_path=db_path,
        max_quarters=max_quarters,
        max_annual=max_annual,
        include_relationship_graph=True,
        return_relationship_graph=True,
    )

    if not docs:
        print(f"No documents generated for {ticker} — skipping index update")
        return None

    if relationship_graph is not None:
        _persist_relationship_graph(relationship_graph, persist_dir=persist_dir)

    # Step 3: upsert into vector store
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        index.insert_nodes(nodes)
        print(f"Updated existing index for {ticker}")
    except FileNotFoundError:
        index = VectorStoreIndex(nodes)
        print(f"Created new index for {ticker}")

    index.storage_context.persist(persist_dir=persist_dir)
    print(f"Index successfully refreshed for {ticker}")


#refresh_ticker_data_and_index('aapl')
