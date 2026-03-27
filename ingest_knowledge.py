import io
import os
import re
import shutil
import stat
import subprocess
from urllib.parse import quote_plus

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader
import pandas as pd
import requests

import ingest_macro
import ingest_stock


load_dotenv("config.env")

DEFAULT_GLOSSARY_BASE_DIR = os.getenv("GLOSSARY_BASE_DIR", "./data_store/glossary")
DEFAULT_GLOSSARY_METADATA_PATH = os.getenv(
    "GLOSSARY_METADATA_PATH",
    "C:/Users/johnk/OneDrive/Desktop/New Text Document.txt",
)
DEFAULT_KNOWLEDGE_STORAGE_DIR = os.getenv("KNOWLEDGE_STORAGE_DIR", "./storage/knowledge")
CFI_GLOSSARY_URL = "https://corporatefinanceinstitute.com/resources/accounting/financial-analysis-ratios-glossary/"
CFA_FORMULA_PDF_URL = "https://www.cfainstitute.org/sites/default/files/-/media/documents/support/programs/cfa/cfa_program_level_ii_financial_ratio_list.pdf"
CFI_CHEAT_SHEET_PDF_URL = "https://corporatefinanceinstitute.com/assets/CFI-Financial-Ratios-Cheat-Sheet-eBook.pdf"
NYU_DEFINITIONS_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/definitions.html"
DEFAULT_COMPANY_RAW_GLOSSARY_DIR = os.path.join(DEFAULT_GLOSSARY_BASE_DIR, "company", "raw")
DEFAULT_ECO_RAW_GLOSSARY_DIR = os.path.join(DEFAULT_GLOSSARY_BASE_DIR, "eco", "raw")
DEFAULT_RAW_GLOSSARY_DIR = DEFAULT_COMPANY_RAW_GLOSSARY_DIR

_HTTP_SESSION = None
_CFI_GLOSSARY_CACHE = None
_CFA_FORMULA_CACHE = None
_CFI_CHEAT_SHEET_CACHE = None
_NYU_DEFINITIONS_CACHE = None
_CFI_RESOURCE_TEXT_CACHE = {}

MANUAL_SOURCE_ALIASES = {
    "gross margin": {"Gross Margin Ratio", "Gross Profit Margin"},
    "operating margin": {"Operating Profit Margin", "EBIT Margin"},
    "net profit margin": {"Net Profit Margin Ratio", "Net Margin"},
    "ebitda margin": {"EBITDA Margin"},
    "return on equity roe": {"Return on Equity", "ROE"},
    "return on assets roa": {"Return on Assets", "ROA"},
    "return on invested capital roic": {"Return on Invested Capital", "ROIC"},
    "return on capital employed roce": {"Return on Capital Employed", "ROCE"},
    "current ratio": {"Current Ratio"},
    "working capital ratio": {"Current Ratio", "Working Capital Ratio"},
    "quick ratio": {"Quick Ratio", "Acid Test"},
    "cash ratio": {"Cash Ratio"},
    "debt to equity d e": {"Debt-to-Equity Ratio", "Debt to Equity Ratio"},
    "net debt ebitda": {"Net Debt/EBITDA", "EV/EBITDA"},
    "asset turnover": {"Asset Turnover Ratio", "Total Asset Turnover Ratio"},
    "inventory turnover": {"Inventory Turnover Ratio"},
    "receivables turnover": {"Receivables Turnover Ratio", "Accounts Receivable Turnover Ratio"},
    "payables turnover": {"Payables Turnover Ratio", "Accounts Payable Turnover"},
    "cash conversion cycle": {"Cash Conversion Cycle", "Cash Conversion Cycle (net operating cycle)"},
    "fixed asset turnover": {"Fixed Asset Turnover Ratio"},
    "interest coverage ratio": {"Interest Coverage Ratio"},
    "interest coverage reit": {"Interest Coverage Ratio"},
    "long term debt to capital": {"Debt-to-Capital Ratio"},
    "dividend payout ratio": {"Dividend Payout Ratio"},
    "dividend yield": {"Dividend Yield Formula", "Dividend Yield"},
    "price to book p b": {"Market to Book Ratio", "Price to Book Ratio", "Price/Book Ratio", "P/B Ratio"},
    "organic revenue growth": {"Organic Growth", "Organic Revenue Growth"},
    "price to earnings p e trailing": {"Price Earnings Ratio", "Price to Earnings Ratio", "P/E Ratio"},
    "price to earnings p e forward": {"Forward P/E Ratio", "Forward PE Ratio", "P/E Ratio"},
    "price to sales p s": {"Price to Sales Ratio", "Price/Sales Ratio", "P/S Ratio"},
    "price to free cash flow p fcf": {"Price to Free Cash Flow", "P/FCF"},
    "peg ratio": {"PEG Ratio"},
    "ev ebitda": {"EV/EBITDA"},
    "ev revenue": {"EV/Revenue", "Enterprise Value to Revenue"},
    "free cash flow margin": {"Free Cash Flow Margin"},
    "operating cash flow margin": {"Operating Cash Flow Ratio", "Operating Cash Flow"},
    "free cash flow yield": {"Free Cash Flow Yield"},
    "share repurchase yield": {"Buyback Yield", "Share Repurchase"},
    "reinvestment rate": {"Reinvestment Rate"},
    "revenue growth yoy": {"Revenue Growth"},
    "quarterly revenue growth": {"Quarterly Revenue Growth", "Quarter over Quarter Growth"},
    "revenue cagr 3 year": {"Compound Annual Growth Rate", "CAGR"},
    "eps growth yoy": {"YoY (Year over Year)", "Earnings Per Share (EPS)"},
    "eps cagr 3 year": {"EPS Growth", "Compound Annual Growth Rate", "CAGR"},
    "free cash flow growth": {"Free Cash Flow"},
    "fcf net income": {"Free Cash Flow Conversion", "Free Cash Flow"},
    "capex depreciation": {"Capital Expenditure", "CapEx", "Growth Capex"},
    "cash flow return on investment cfroi": {"Cash Flow Return on Investment", "CFROI"},
    "revenue growth yoy": {"YoY (Year over Year)"},
    "reinvestment rate": {"Retention Ratio"},
    "cap rate": {"Capitalization Rate", "Cap Rate"},
}

CFI_APPEND_START = "[CFI_WEBSITE_APPEND_START]"
CFI_APPEND_END = "[CFI_WEBSITE_APPEND_END]"

CFI_DIRECT_RESOURCE_OVERRIDES = {
    "eps growth yoy": (
        "YoY (Year over Year)",
        "https://corporatefinanceinstitute.com/resources/accounting/year-over-year-yoy-analysis/",
    ),
    "revenue growth yoy": (
        "YoY (Year over Year)",
        "https://corporatefinanceinstitute.com/resources/accounting/year-over-year-yoy-analysis/",
    ),
}

CFI_MACRO_RESOURCE_OVERRIDES = {
    "fed_funds_rate": [
        ("Federal Funds Rate", "https://corporatefinanceinstitute.com/resources/economics/federal-funds-rate/"),
    ],
    "real_gdp": [
        ("Nominal GDP vs. Real GDP", "https://corporatefinanceinstitute.com/resources/economics/nominal-real-gdp/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
    "real_gdp_qoq_pct": [
        ("Nominal GDP vs. Real GDP", "https://corporatefinanceinstitute.com/resources/economics/nominal-real-gdp/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
    "cpi_all_items": [
        ("Consumer Price Index (CPI)", "https://corporatefinanceinstitute.com/resources/economics/consumer-price-index-cpi/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
    "cpi_inflation_yoy": [
        ("Consumer Price Index (CPI)", "https://corporatefinanceinstitute.com/resources/economics/consumer-price-index-cpi/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
    "unemployment_rate": [
        ("Unemployment", "https://corporatefinanceinstitute.com/resources/economics/unemployment/"),
        ("Labor Force KPIs", "https://corporatefinanceinstitute.com/resources/economics/labor-force-kpis/"),
    ],
    "adp_private_payrolls": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
        ("Labor Force KPIs", "https://corporatefinanceinstitute.com/resources/economics/labor-force-kpis/"),
    ],
    "adp_private_payrolls_mom_change": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
        ("Labor Force KPIs", "https://corporatefinanceinstitute.com/resources/economics/labor-force-kpis/"),
    ],
    "nonfarm_payrolls": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
    ],
    "nonfarm_payrolls_mom_change": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
    ],
    "average_hourly_earnings": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
        ("Labor Force KPIs", "https://corporatefinanceinstitute.com/resources/economics/labor-force-kpis/"),
    ],
    "average_hourly_earnings_yoy": [
        ("Non-Farm Payroll", "https://corporatefinanceinstitute.com/resources/economics/non-farm-payroll/"),
        ("Labor Force KPIs", "https://corporatefinanceinstitute.com/resources/economics/labor-force-kpis/"),
    ],
    "ism_manufacturing_pmi": [
        ("ISM Manufacturing Index", "https://corporatefinanceinstitute.com/resources/economics/ism-manufacturing-index/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
    "ism_services_pmi": [
        ("ISM Manufacturing Index", "https://corporatefinanceinstitute.com/resources/economics/ism-manufacturing-index/"),
        ("Economic Indicators", "https://corporatefinanceinstitute.com/resources/economics/economic-indicators/"),
    ],
}



def _clean_text_field(value):
    if value is None:
        return None
    cleaned_value = str(value).strip()
    return cleaned_value or None


def _normalize_source_text(text):
    cleaned_text = str(text or "")
    replacements = {
        "\xa0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00f7": "/",
        "\u00d7": "x",
        "\u2044": "/",
        "\xad": "",
    }
    for source_char, target_char in replacements.items():
        cleaned_text = cleaned_text.replace(source_char, target_char)
    return cleaned_text


def _indicator_name_from_path(glossary_path):
    indicator_name = os.path.splitext(os.path.basename(glossary_path))[0]
    return indicator_name.replace("_", " ").strip()


def _extract_glossary_header_value(text, header_name):
    target_prefix = f"{header_name.lower()}:"
    for line in text.splitlines():
        normalized_line = line.lstrip("\ufeff").strip()
        if normalized_line.lower().startswith(target_prefix):
            return _clean_text_field(normalized_line.split(":", 1)[1])
    return None


def _normalize_indicator_key(value):
    normalized_value = re.sub(r"[^a-z0-9]+", " ", _normalize_source_text(value).lower())
    return re.sub(r"\s+", " ", normalized_value).strip()


def _indicator_aliases(indicator_name):
    aliases = {_clean_text_field(indicator_name) or ""}
    base_name = re.sub(r"\s*\([^)]*\)", "", indicator_name or "").strip()
    if base_name:
        aliases.add(base_name)

    for acronym in re.findall(r"\(([^)]+)\)", indicator_name or ""):
        cleaned_acronym = _clean_text_field(acronym)
        if cleaned_acronym:
            aliases.add(cleaned_acronym)

    normalized_variants = {
        alias.replace("-", " ").replace("/", " ").replace("%", " percent")
        for alias in aliases
        if alias
    }
    aliases.update(alias.strip() for alias in normalized_variants if alias.strip())
    return {alias for alias in aliases if alias}


def _http_session():
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        _HTTP_SESSION = session
    return _HTTP_SESSION


def _http_get_text(url, timeout=60):
    response = _http_session().get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return _normalize_source_text(response.text)


def _http_get_bytes(url, timeout=60):
    response = _http_session().get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _clean_block_text(text):
    normalized_text = _normalize_source_text(text)
    normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
    return normalized_text.strip()


def _load_cfi_glossary_entries():
    global _CFI_GLOSSARY_CACHE
    if _CFI_GLOSSARY_CACHE is not None:
        return _CFI_GLOSSARY_CACHE

    soup = BeautifulSoup(_http_get_text(CFI_GLOSSARY_URL), "html.parser")
    entries = {}
    for heading in soup.find_all("h6"):
        title = _clean_block_text(heading.get_text(" ", strip=True))
        if not title:
            continue

        paragraphs = []
        sibling = heading.find_next_sibling()
        while sibling is not None and getattr(sibling, "name", None) != "h6":
            if getattr(sibling, "name", None) == "p":
                paragraph_text = _clean_block_text(sibling.get_text(" ", strip=True))
                if paragraph_text:
                    paragraphs.append(paragraph_text)
            sibling = sibling.find_next_sibling()

        if paragraphs:
            entries[title] = "\n\n".join(paragraphs)

    _CFI_GLOSSARY_CACHE = entries
    return _CFI_GLOSSARY_CACHE


def _load_cfa_formula_entries():
    global _CFA_FORMULA_CACHE
    if _CFA_FORMULA_CACHE is not None:
        return _CFA_FORMULA_CACHE

    reader = PdfReader(io.BytesIO(_http_get_bytes(CFA_FORMULA_PDF_URL)))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    extracted_text = _normalize_source_text(extracted_text)
    lines = []
    for raw_line in extracted_text.splitlines():
        clean_line = _clean_block_text(raw_line)
        if not clean_line:
            continue
        if clean_line in {"i", "ii"} or clean_line == "FINANCIAL RATIO LIST":
            continue
        lines.append(clean_line)

    entries = []
    current_entry = ""
    for line in lines:
        if re.match(r"^\d+\s+", line):
            if current_entry:
                entries.append(current_entry.strip())
            current_entry = re.sub(r"^\d+\s+", "", line).strip()
        elif current_entry:
            current_entry = f"{current_entry} {line}".strip()
    if current_entry:
        entries.append(current_entry.strip())

    formulas = {}
    for entry in entries:
        normalized_entry = _clean_block_text(entry)
        if "=" not in normalized_entry:
            continue
        ratio_name, formula = normalized_entry.split("=", 1)
        ratio_name = ratio_name.strip()
        formula = formula.strip()
        if ratio_name and formula:
            formulas[ratio_name] = formula

    _CFA_FORMULA_CACHE = formulas
    return _CFA_FORMULA_CACHE


def _extract_between_markers(text, start_marker, end_marker=None):
    if not text:
        return ""

    start_match = re.search(re.escape(start_marker), text, flags=re.IGNORECASE)
    if not start_match:
        return ""

    start_index = start_match.end()
    end_index = len(text)
    if end_marker:
        end_match = re.search(re.escape(end_marker), text[start_index:], flags=re.IGNORECASE)
        if end_match:
            end_index = start_index + end_match.start()

    return _clean_block_text(text[start_index:end_index])


def _clean_pdf_lines(page_text):
    cleaned_lines = []
    for raw_line in _normalize_source_text(page_text).splitlines():
        clean_line = _clean_block_text(raw_line)
        if not clean_line:
            continue
        lowered_line = clean_line.lower()
        if lowered_line.startswith("corporate finance institute"):
            continue
        if lowered_line.startswith("corporatefinanceinstitute.com"):
            continue
        if lowered_line.startswith("the corporate finance institute"):
            continue
        if clean_line.isdigit():
            continue
        cleaned_lines.append(clean_line)
    return cleaned_lines


def _load_cfi_cheat_sheet_entries():
    global _CFI_CHEAT_SHEET_CACHE
    if _CFI_CHEAT_SHEET_CACHE is not None:
        return _CFI_CHEAT_SHEET_CACHE

    reader = PdfReader(io.BytesIO(_http_get_bytes(CFI_CHEAT_SHEET_PDF_URL)))
    ignored_titles = {
        "Table of Contents",
        "Financial Ratio Analysis Overview",
        "Profitability Ratio",
        "Leverage Ratios",
        "Efficiency Ratios",
        "Liquidity Ratios",
        "Cash Flow Indicator Ratios",
        "Investment Valuation Ratios",
    }

    entries = {}
    for page in reader.pages:
        lines = _clean_pdf_lines(page.extract_text() or "")
        if not lines:
            continue

        title = lines[0]
        if title in ignored_titles or title.endswith("Ratios"):
            continue

        body = "\n".join(lines[1:])
        if "Overview" not in body and "Interpretation" not in body:
            continue

        overview_text = _extract_between_markers(body, "Overview", "Formula")
        interpretation_text = _extract_between_markers(body, "Interpretation")

        sections = []
        if overview_text:
            sections.append(f"Overview:\n{overview_text}")
        if interpretation_text:
            sections.append(f"Interpretation:\n{interpretation_text}")

        if sections:
            entries[title] = "\n\n".join(sections)

    _CFI_CHEAT_SHEET_CACHE = entries
    return _CFI_CHEAT_SHEET_CACHE


def _load_nyu_definitions_entries():
    global _NYU_DEFINITIONS_CACHE
    if _NYU_DEFINITIONS_CACHE is not None:
        return _NYU_DEFINITIONS_CACHE

    table = pd.read_html(NYU_DEFINITIONS_URL)[0]
    table.columns = [_clean_block_text(value) for value in table.iloc[0]]
    table = table.iloc[1:].copy()

    entries = {}
    for _, row in table.iterrows():
        variable_name = _clean_block_text(row.get("Variables"))
        if not variable_name or variable_name.lower() == "nan":
            continue

        measure_text = _clean_block_text(row.get("What it tries to measure"))
        comment_text = _clean_block_text(row.get("Comments"))
        entries[variable_name] = {
            "what_it_tries_to_measure": "" if measure_text.lower() == "nan" else measure_text,
            "comment": "" if comment_text.lower() == "nan" else comment_text,
        }

    _NYU_DEFINITIONS_CACHE = entries
    return _NYU_DEFINITIONS_CACHE


def _source_alias_candidates(indicator_name):
    candidates = set(_indicator_aliases(indicator_name))
    normalized_name = _normalize_indicator_key(indicator_name)
    for alias in MANUAL_SOURCE_ALIASES.get(normalized_name, set()):
        cleaned_alias = _clean_text_field(alias)
        if cleaned_alias:
            candidates.add(cleaned_alias)
    return {candidate for candidate in candidates if candidate}


def _lookup_source_match(indicator_name, source_entries):
    if not source_entries:
        return None

    normalized_source_map = {
        _normalize_indicator_key(source_name): (source_name, source_value)
        for source_name, source_value in source_entries.items()
        if _normalize_indicator_key(source_name)
    }

    for alias in _source_alias_candidates(indicator_name):
        normalized_alias = _normalize_indicator_key(alias)
        if normalized_alias in normalized_source_map:
            return normalized_source_map[normalized_alias]

    return None


def _search_cfi_resource_links(query_text):
    search_url = f"https://corporatefinanceinstitute.com/?s={quote_plus(query_text)}"
    try:
        soup = BeautifulSoup(_http_get_text(search_url), "html.parser")
    except requests.RequestException:
        return []
    candidates = []
    for card in soup.select("div.page-card a[href]"):
        href = _clean_text_field(card.get("href"))
        title = _clean_text_field(card.get("data-title") or card.get_text(" ", strip=True))
        if not href or not title:
            continue
        if "corporatefinanceinstitute.com/resources/" not in href:
            continue
        lowered_href = href.lower()
        if "calculator" in lowered_href or "template" in lowered_href:
            continue
        candidates.append((title, href))
    return candidates


def _score_cfi_candidate(indicator_name, candidate_title, candidate_url):
    indicator_terms = {
        _normalize_indicator_key(alias)
        for alias in _source_alias_candidates(indicator_name)
    }
    indicator_terms = {term for term in indicator_terms if term}
    normalized_title = _normalize_indicator_key(candidate_title)
    normalized_url = _normalize_indicator_key(candidate_url)

    qualifier_groups = [
        {"yoy", "year over year"},
        {"quarterly", "qoq", "quarter over quarter"},
        {"cagr", "compound annual growth rate"},
        {"forward"},
        {"trailing", "ttm"},
    ]
    candidate_text = f"{normalized_title} {normalized_url}".strip()
    indicator_text = _normalize_indicator_key(indicator_name)
    for qualifier_group in qualifier_groups:
        indicator_has_qualifier = any(qualifier in indicator_text for qualifier in qualifier_group)
        candidate_has_qualifier = any(qualifier in candidate_text for qualifier in qualifier_group)
        if indicator_has_qualifier and not candidate_has_qualifier:
            return -100

    score = 0
    for term in indicator_terms:
        if term == normalized_title:
            score += 10
        elif term and term in normalized_title:
            score += 6
        elif term and term in normalized_url:
            score += 3
        elif normalized_title and normalized_title in term:
            score += 2

    title_words = set(normalized_title.split())
    for term in indicator_terms:
        score += len(title_words & set(term.split()))

    return score


def _find_cfi_resource_page(indicator_name):
    direct_override = CFI_DIRECT_RESOURCE_OVERRIDES.get(_normalize_indicator_key(indicator_name))
    if direct_override:
        return direct_override

    seen_links = {}
    for query_text in [indicator_name, *_source_alias_candidates(indicator_name)]:
        for title, href in _search_cfi_resource_links(query_text):
            seen_links[href] = title

    best_choice = None
    best_score = 0
    for href, title in seen_links.items():
        score = _score_cfi_candidate(indicator_name, title, href)
        if score > best_score:
            best_score = score
            best_choice = (title, href)

    if best_score < 3:
        return None
    return best_choice


def _extract_cfi_resource_text(resource_url):
    cached_text = _CFI_RESOURCE_TEXT_CACHE.get(resource_url)
    if cached_text is not None:
        return cached_text

    soup = BeautifulSoup(_http_get_text(resource_url), "html.parser")
    heading = soup.find("h1")
    if not heading:
        return ""

    collected_lines = []
    for element in heading.find_all_next():
        if not isinstance(element, Tag):
            continue

        if element.name in {"h2", "h3"}:
            section_heading = _clean_block_text(element.get_text(" ", strip=True))
            lowered_heading = section_heading.lower()
            if lowered_heading.startswith("additional resources"):
                break
            if lowered_heading.startswith("related readings"):
                break
            if lowered_heading.startswith("create a free account"):
                break
            if lowered_heading.startswith("supercharge your skills"):
                break
            if lowered_heading.startswith("access exclusive templates"):
                break
            collected_lines.append(f"## {section_heading}")
            continue

        if element.name == "p":
            paragraph_text = _clean_block_text(element.get_text(" ", strip=True))
            lowered_paragraph = paragraph_text.lower()
            if lowered_paragraph.startswith("share this article"):
                break
            if paragraph_text:
                collected_lines.append(paragraph_text)
            continue

        if element.name == "li":
            list_text = _clean_block_text(element.get_text(" ", strip=True))
            if list_text:
                collected_lines.append(f"- {list_text}")

    cleaned_lines = []
    seen_lines = set()
    for line in collected_lines:
        normalized_line = _normalize_indicator_key(line)
        if not normalized_line or normalized_line in seen_lines:
            continue
        seen_lines.add(normalized_line)
        cleaned_lines.append(line)

    resource_text = "\n".join(cleaned_lines).strip()
    resource_text = re.split(r"\n## Related Readings\b", resource_text, maxsplit=1)[0].strip()
    resource_text = re.split(r"\nShare this article\b", resource_text, maxsplit=1)[0].strip()
    _CFI_RESOURCE_TEXT_CACHE[resource_url] = resource_text
    return resource_text


def _split_raw_glossary_sections(text):
    sections = {}
    section_patterns = [
        ("header", "Section 1: Definition"),
        ("definition", "Section 2: Formula"),
        ("formula", "Section 3: Additional Information"),
        ("additional", "Section 4: What It Tries To Measure"),
        ("measure", "Section 5: Comment"),
        ("comment", None),
    ]

    current_text = text
    for section_name, next_marker in section_patterns:
        if next_marker is None:
            sections[section_name] = current_text
            break

        before, marker, after = current_text.partition(next_marker)
        sections[section_name] = before
        current_text = next_marker + after

    return sections


def append_cfi_site_information_to_raw_files(raw_glossary_dir=DEFAULT_RAW_GLOSSARY_DIR):
    raw_dir = os.path.abspath(raw_glossary_dir)
    if not os.path.isdir(raw_dir):
        return []

    updated_files = []
    for entry in sorted(os.scandir(raw_dir), key=lambda item: item.name.lower()):
        if not entry.is_file() or not entry.name.lower().endswith(".md"):
            continue

        with open(entry.path, "r", encoding="utf-8") as raw_file:
            original_text = raw_file.read()

        indicator_name = None
        for line in original_text.splitlines():
            if line.startswith("Indicator: "):
                indicator_name = _clean_text_field(line.split(":", 1)[1])
                break

        if not indicator_name:
            continue

        resource_match = _find_cfi_resource_page(indicator_name)
        appended_block = ""
        if resource_match:
            resource_title, resource_url = resource_match
            resource_text = _extract_cfi_resource_text(resource_url)
            if resource_text:
                appended_block = (
                    f"{CFI_APPEND_START}\n"
                    f"CFI Website Resource: {resource_title}\n"
                    f"CFI Website URL: {resource_url}\n\n"
                    f"{resource_text}\n"
                    f"{CFI_APPEND_END}"
                )

        definition_marker = "Section 1: Definition\n"
        formula_marker = "\n\nSection 2: Formula\n"
        additional_marker = "\n\nSection 3: Additional Information\n"
        measure_marker = "\n\nSection 4: What It Tries To Measure\n"
        comment_marker = "\n\nSection 5: Comment\n"

        if not all(marker in original_text for marker in [
            definition_marker,
            formula_marker,
            additional_marker,
            measure_marker,
            comment_marker,
        ]):
            continue

        before_additional, _, after_additional = original_text.partition(additional_marker)
        additional_body, _, trailing_text = after_additional.partition(measure_marker)

        base_additional = re.sub(
            rf"\n*{re.escape(CFI_APPEND_START)}.*?{re.escape(CFI_APPEND_END)}\n*",
            "\n\n",
            additional_body,
            flags=re.DOTALL,
        ).strip()

        new_additional_parts = []
        if base_additional:
            new_additional_parts.append(base_additional)
        if appended_block:
            new_additional_parts.append(appended_block)

        new_additional_body = "\n\n".join(part.strip() for part in new_additional_parts if part.strip())
        updated_text = (
            before_additional
            + additional_marker
            + new_additional_body
            + measure_marker
            + trailing_text
        )

        if updated_text != original_text:
            with open(entry.path, "w", encoding="utf-8") as raw_file:
                raw_file.write(updated_text)
            updated_files.append(entry.path)

    return updated_files


def _raw_glossary_file_name(indicator_name):
    safe_name = _normalize_source_text(indicator_name)
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", safe_name)
    safe_name = safe_name.strip("_")
    return f"{safe_name}.md"


def _section_text(value):
    cleaned_value = _clean_block_text(value)
    return cleaned_value if cleaned_value else ""


def _glossary_markdown_directories(glossary_base_dir):
    candidate_dirs = [
        os.path.join(glossary_base_dir, "company", "raw"),
        os.path.join(glossary_base_dir, "company"),
        os.path.join(glossary_base_dir, "eco", "raw"),
        os.path.join(glossary_base_dir, "eco"),
        glossary_base_dir,
    ]
    discovered_dirs = []
    seen_dirs = set()

    for candidate_dir in candidate_dirs:
        normalized_dir = os.path.normpath(candidate_dir)
        if normalized_dir in seen_dirs or not os.path.isdir(candidate_dir):
            continue

        glossary_files = [
            entry.name
            for entry in os.scandir(candidate_dir)
            if entry.is_file() and entry.name.lower().endswith((".md", ".txt"))
        ]
        if not glossary_files:
            continue

        seen_dirs.add(normalized_dir)
        discovered_dirs.append(candidate_dir)

    return discovered_dirs


def build_raw_glossary_files(
    metadata_path=DEFAULT_GLOSSARY_METADATA_PATH,
    raw_glossary_dir=DEFAULT_RAW_GLOSSARY_DIR,
):
    metadata_records = _load_indicator_metadata(metadata_path=metadata_path)
    os.makedirs(raw_glossary_dir, exist_ok=True)

    cfi_glossary_entries = _load_cfi_glossary_entries()
    cfa_formula_entries = _load_cfa_formula_entries()
    cfi_cheat_sheet_entries = _load_cfi_cheat_sheet_entries()
    nyu_entries = _load_nyu_definitions_entries()

    generated_files = []
    for record in metadata_records:
        indicator_name = record["name"]
        definition_match = _lookup_source_match(indicator_name, cfi_glossary_entries)
        formula_match = _lookup_source_match(indicator_name, cfa_formula_entries)
        additional_info_match = _lookup_source_match(indicator_name, cfi_cheat_sheet_entries)
        nyu_match = _lookup_source_match(indicator_name, nyu_entries)

        definition_text = _section_text(definition_match[1] if definition_match else "")
        formula_text = _section_text(formula_match[1] if formula_match else "")
        additional_info_text = _section_text(additional_info_match[1] if additional_info_match else "")
        measure_text = _section_text((nyu_match[1] or {}).get("what_it_tries_to_measure", "") if nyu_match else "")
        comment_text = _section_text((nyu_match[1] or {}).get("comment", "") if nyu_match else "")

        raw_text = (
            f"Indicator: {indicator_name}\n"
            f"Group: {record['group']}\n"
            f"Subgroup: {record['subgroup']}\n\n"
            "Section 1: Definition\n"
            f"{definition_text}\n\n"
            "Section 2: Formula\n"
            f"{formula_text}\n\n"
            "Section 3: Additional Information\n"
            f"{additional_info_text}\n\n"
            "Section 4: What It Tries To Measure\n"
            f"{measure_text}\n\n"
            "Section 5: Comment\n"
            f"{comment_text}\n"
        )

        output_path = os.path.join(raw_glossary_dir, _raw_glossary_file_name(indicator_name))
        with open(output_path, "w", encoding="utf-8") as raw_file:
            raw_file.write(raw_text)
        generated_files.append(output_path)

    return generated_files


def _macro_indicator_records():
    records = []
    for series_def in ingest_macro.FRED_SERIES_CATALOG:
        records.append(
            {
                "indicator_key": series_def["indicator_key"],
                "indicator_name": series_def["indicator_name"],
                "group": "macro",
                "subgroup": series_def["category"],
            }
        )
        for derivation in series_def.get("derivations", []):
            records.append(
                {
                    "indicator_key": derivation["indicator_key"],
                    "indicator_name": derivation["indicator_name"],
                    "group": "macro",
                    "subgroup": series_def["category"],
                }
            )

    for series_def in ingest_macro.ISM_SERIES_CATALOG:
        records.append(
            {
                "indicator_key": series_def["indicator_key"],
                "indicator_name": series_def["indicator_name"],
                "group": "macro",
                "subgroup": series_def["category"],
            }
        )

    return records


def _macro_resource_bundle(indicator_key):
    source_entries = CFI_MACRO_RESOURCE_OVERRIDES.get(indicator_key, [])
    resource_sections = []
    seen_urls = set()

    for source_title, source_url in source_entries:
        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)

        resource_text = _extract_cfi_resource_text(source_url)
        if not resource_text:
            continue

        resource_sections.append(
            "\n".join(
                [
                    f"CFI Source: {source_title}",
                    f"CFI URL: {source_url}",
                    "",
                    resource_text,
                ]
            ).strip()
        )

    return "\n\n".join(resource_sections).strip()


def build_macro_raw_glossary_files(raw_glossary_dir=DEFAULT_ECO_RAW_GLOSSARY_DIR):
    os.makedirs(raw_glossary_dir, exist_ok=True)
    generated_files = []

    for record in _macro_indicator_records():
        resource_bundle = _macro_resource_bundle(record["indicator_key"])
        raw_text = (
            f"Indicator: {record['indicator_name']}\n"
            f"Group: {record['group']}\n"
            f"Subgroup: {record['subgroup']}\n\n"
            f"{resource_bundle}\n"
        )

        output_path = os.path.join(raw_glossary_dir, _raw_glossary_file_name(record["indicator_name"]))
        with open(output_path, "w", encoding="utf-8") as raw_file:
            raw_file.write(raw_text)
        generated_files.append(output_path)

    return generated_files


def _load_indicator_metadata(metadata_path=DEFAULT_GLOSSARY_METADATA_PATH):
    if not metadata_path or not os.path.isfile(metadata_path):
        return []

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        text = metadata_file.read()

    pattern = re.compile(
        r'\{\s*"name":\s*"(?P<name>[^"]+)"\s*,\s*"group":\s*"(?P<group>[^"]+)"\s*,\s*"subgroup":\s*"(?P<subgroup>[^"]+)"(?:\s*,\s*"category":\s*"(?P<category>[^"]+)")?\s*\}',
        flags=re.MULTILINE,
    )

    records = []
    for match in pattern.finditer(text):
        record = {
            "name": match.group("name").strip(),
            "group": match.group("group").strip(),
            "subgroup": match.group("subgroup").strip(),
        }
        if record["group"] == "industry_specific":
            continue
        record["aliases"] = sorted(_indicator_aliases(record["name"]))
        records.append(record)

    return records


def _resolve_indicator_metadata(indicator_name, metadata_records):
    normalized_indicator = _normalize_indicator_key(indicator_name)
    if not normalized_indicator:
        return None

    exact_match = None
    subset_match = None
    for record in metadata_records:
        normalized_aliases = {
            _normalize_indicator_key(alias)
            for alias in record.get("aliases", [])
            if _normalize_indicator_key(alias)
        }
        if normalized_indicator in normalized_aliases:
            exact_match = record
            break

        if any(
            normalized_indicator == alias
            or normalized_indicator in alias
            or alias in normalized_indicator
            for alias in normalized_aliases
        ):
            subset_match = subset_match or record

    return exact_match or subset_match


def build_glossary_docs(
    glossary_base_dir=DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=DEFAULT_GLOSSARY_METADATA_PATH,
):
    glossary_dirs = _glossary_markdown_directories(glossary_base_dir)
    if not glossary_dirs:
        return []

    metadata_records = _load_indicator_metadata(metadata_path=metadata_path)
    glossary_documents = []
    for glossary_dir in glossary_dirs:
        normalized_glossary_dir = os.path.normpath(glossary_dir)
        glossary_dir_name = os.path.basename(normalized_glossary_dir)
        if glossary_dir_name.lower() == "raw":
            glossary_domain = os.path.basename(os.path.dirname(normalized_glossary_dir)) or "general"
        else:
            glossary_domain = glossary_dir_name or "general"
        for entry in sorted(os.scandir(glossary_dir), key=lambda item: item.name.lower()):
            if not entry.is_file() or not entry.name.lower().endswith((".md", ".txt")):
                continue

            with open(entry.path, "r", encoding="utf-8") as glossary_file:
                text = glossary_file.read().strip()

            if not text:
                continue

            indicator_name = _extract_glossary_header_value(text, "Indicator")
            if not indicator_name:
                continue

            resolved_indicator_name = indicator_name or _indicator_name_from_path(entry.path)
            resolved_metadata = _resolve_indicator_metadata(
                resolved_indicator_name,
                metadata_records,
            )
            file_group = _extract_glossary_header_value(text, "Group")
            file_subgroup = _extract_glossary_header_value(text, "Subgroup")
            default_group = "macro" if glossary_domain == "eco" else "unmapped"
            default_subgroup = "macro" if glossary_domain == "eco" else "unmapped"
            group = (resolved_metadata or {}).get("group") or file_group or default_group
            subgroup = (resolved_metadata or {}).get("subgroup") or file_subgroup or default_subgroup
            canonical_name = (resolved_metadata or {}).get("name", resolved_indicator_name)
            indicator_aliases = sorted(
                _indicator_aliases(canonical_name) | _indicator_aliases(resolved_indicator_name)
            )

            glossary_documents.append(
                Document(
                    text=(
                        f"Indicator Group: {group}\n"
                        f"Indicator Subgroup: {subgroup}\n"
                        f"Glossary Domain: {glossary_domain}\n\n"
                        f"{text}"
                    ),
                    metadata={
                        "type": "indicator_glossary",
                        "indicator_name": resolved_indicator_name,
                        "indicator_canonical_name": canonical_name,
                        "indicator_aliases": indicator_aliases,
                        "group": group,
                        "subgroup": subgroup,
                        "glossary_domain": glossary_domain,
                        "glossary_category": "specific_indicator",
                        "glossary_path": entry.path,
                        "source": "glossary_markdown",
                        "metadata_source_path": metadata_path,
                    },
                )
            )

    return glossary_documents


def get_glossary_indicator_catalog(
    glossary_base_dir=DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=DEFAULT_GLOSSARY_METADATA_PATH,
    grouped=True,
):
    indicator_records = []
    seen_indicators = set()

    for doc in build_glossary_docs(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    ):
        metadata = doc.metadata or {}
        indicator_name = (
            metadata.get("indicator_canonical_name")
            or metadata.get("indicator_name")
            or ""
        ).strip()
        if not indicator_name:
            continue

        normalized_name = _normalize_indicator_key(indicator_name)
        if normalized_name in seen_indicators:
            continue
        seen_indicators.add(normalized_name)

        aliases = metadata.get("indicator_aliases") or []
        indicator_records.append(
            {
                "indicator_name": indicator_name,
                "group": (metadata.get("group") or "unmapped").strip(),
                "subgroup": (metadata.get("subgroup") or "unmapped").strip(),
                "aliases": sorted({alias for alias in aliases if alias}),
                "glossary_path": metadata.get("glossary_path"),
            }
        )

    indicator_records.sort(
        key=lambda record: (
            record["group"].lower(),
            record["subgroup"].lower(),
            record["indicator_name"].lower(),
        )
    )

    if not grouped:
        return indicator_records

    catalog = {}
    for record in indicator_records:
        group_bucket = catalog.setdefault(record["group"], {})
        subgroup_bucket = group_bucket.setdefault(record["subgroup"], [])
        subgroup_bucket.append(record["indicator_name"])

    return catalog


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
            try:
                os.rmdir(persist_dir)
            except OSError:
                shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
        else:
            shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
    os.makedirs(persist_dir, exist_ok=True)


def refresh_knowledge_index(
    glossary_base_dir=DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=DEFAULT_GLOSSARY_METADATA_PATH,
    storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
):
    ingest_stock.env()

    docs = build_glossary_docs(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    if not docs:
        print("No glossary documents found; skipping knowledge index refresh.")
        return None

    os.makedirs(os.path.dirname(os.path.normpath(storage_dir)) or ".", exist_ok=True)
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)
    _reset_persist_dir(storage_dir)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Knowledge index successfully refreshed at {storage_dir}")
    return storage_dir
