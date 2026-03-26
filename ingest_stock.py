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
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE = (
        "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dashes}/{primary_document}"
    )
DEFAULT_SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
_SEC_TICKER_MAP = None

IDENTIFIER_COLUMNS = ["Ticker", "Frequency", "Period End Date"]
RAW_FACT_COLUMNS = [
    "Total Revenue",
    "Cost of Revenue",
    "Gross Profit",
    "Operating Income",
    "EBITDA",
    "Net Income",
    "Income Before Tax",
    "Income Tax Expense",
    "Diluted EPS",
    "Total Assets",
    "Total Liabilities",
    "Shareholders Equity",
    "Current Assets",
    "Current Liabilities",
    "Cash and Equivalents",
    "Inventory",
    "Accounts Receivable",
    "Accounts Payable",
    "PPE Net",
    "Current Debt",
    "Long-term Debt",
    "Total Debt",
    "Interest Expense",
    "Operating Cash Flow",
    "Capital Expenditures",
    "Free Cash Flow",
    "Depreciation & Amortization",
    "Diluted Shares",
    "Shares Outstanding",
    "Dividends Paid Common",
    "Dividends Per Share",
    "Share Repurchases",
    "Current Market Price",
    "Current Market Capitalization",
    "Current Enterprise Value",
]
CORE_GLOSSARY_INDICATORS = [
    "Gross Margin",
    "Operating Margin",
    "Net Profit Margin",
    "EBITDA Margin",
    "Return on Equity (ROE)",
    "Return on Assets (ROA)",
    "Return on Invested Capital (ROIC)",
    "Revenue Growth YoY",
    "Quarterly Revenue Growth",
    "Revenue CAGR 3-Year",
    "EPS Growth YoY",
    "EPS CAGR 3-Year",
    "Free Cash Flow Growth",
    "Organic Revenue Growth",
    "Price-to-Earnings (P/E) Trailing",
    "Price-to-Earnings (P/E) Forward",
    "PEG Ratio",
    "Price-to-Book (P/B)",
    "Price-to-Sales (P/S)",
    "EV/EBITDA",
    "EV/Revenue",
    "Price-to-Free Cash Flow (P/FCF)",
    "Current Ratio",
    "Quick Ratio",
    "Cash Ratio",
    "Working Capital Ratio",
    "Debt-to-Equity (D/E)",
    "Net Debt/EBITDA",
    "Interest Coverage Ratio",
    "Debt Service Coverage Ratio",
    "Long-term Debt to Capital",
    "Asset Turnover",
    "Inventory Turnover",
    "Receivables Turnover",
    "Payables Turnover",
    "Cash Conversion Cycle",
    "Operating Cycle",
    "Fixed Asset Turnover",
    "Free Cash Flow Yield",
    "FCF / Net Income",
    "CapEx / Depreciation",
    "Share Repurchase Yield",
    "Dividend Payout Ratio",
    "Dividend Yield",
    "Reinvestment Rate",
    "Operating Cash Flow Margin",
    "Free Cash Flow Margin",
    "Cash Flow Return on Investment (CFROI)",
]
FINANCIAL_INDICATOR_COLUMN_ORDER = IDENTIFIER_COLUMNS + RAW_FACT_COLUMNS + CORE_GLOSSARY_INDICATORS
SEC_FACT_SPECS = {
    "Total Revenue": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "Revenues"),
        ],
        "units": ["USD"],
    },
    "Cost of Revenue": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "CostOfGoodsSold"),
            ("us-gaap", "CostOfRevenue"),
            ("us-gaap", "CostOfGoodsAndServicesSold"),
            ("us-gaap", "CostOfSales"),
        ],
        "units": ["USD"],
    },
    "Gross Profit": {
        "kind": "duration",
        "concepts": [("us-gaap", "GrossProfit")],
        "units": ["USD"],
    },
    "Operating Income": {
        "kind": "duration",
        "concepts": [("us-gaap", "OperatingIncomeLoss")],
        "units": ["USD"],
    },
    "Net Income": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
        ],
        "units": ["USD"],
    },
    "Income Before Tax": {
        "kind": "duration",
        "concepts": [("us-gaap", "IncomeBeforeTaxExpenseBenefit")],
        "units": ["USD"],
    },
    "Income Tax Expense": {
        "kind": "duration",
        "concepts": [("us-gaap", "IncomeTaxExpenseBenefit")],
        "units": ["USD"],
    },
    "Diluted EPS": {
        "kind": "duration",
        "concepts": [("us-gaap", "EarningsPerShareDiluted")],
        "units": ["USD/shares"],
    },
    "Total Assets": {
        "kind": "instant",
        "concepts": [("us-gaap", "Assets")],
        "units": ["USD"],
    },
    "Total Liabilities": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "Liabilities"),
        ],
        "units": ["USD"],
    },
    "Shareholders Equity": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ("us-gaap", "CommonStockholdersEquity"),
        ],
        "units": ["USD"],
    },
    "Current Assets": {
        "kind": "instant",
        "concepts": [("us-gaap", "AssetsCurrent")],
        "units": ["USD"],
    },
    "Current Liabilities": {
        "kind": "instant",
        "concepts": [("us-gaap", "LiabilitiesCurrent")],
        "units": ["USD"],
    },
    "Cash and Equivalents": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
            ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ],
        "units": ["USD"],
    },
    "Inventory": {
        "kind": "instant",
        "concepts": [("us-gaap", "InventoryNet")],
        "units": ["USD"],
    },
    "Accounts Receivable": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "AccountsReceivableNetCurrent"),
            ("us-gaap", "ReceivablesNetCurrent"),
        ],
        "units": ["USD"],
    },
    "Accounts Payable": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "AccountsPayableCurrent"),
            ("us-gaap", "AccountsPayable"),
        ],
        "units": ["USD"],
    },
    "PPE Net": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "PropertyPlantAndEquipmentNet"),
            ("us-gaap", "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"),
        ],
        "units": ["USD"],
    },
    "Current Debt": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "LongTermDebtCurrent"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligationsCurrent"),
            ("us-gaap", "ShortTermBorrowings"),
            ("us-gaap", "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"),
        ],
        "units": ["USD"],
    },
    "Long-term Debt": {
        "kind": "instant",
        "concepts": [
            ("us-gaap", "LongTermDebtNoncurrent"),
            ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
            ("us-gaap", "LongTermDebtAndFinanceLeaseObligations"),
        ],
        "units": ["USD"],
    },
    "Interest Expense": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "InterestExpenseAndDebtExpense"),
            ("us-gaap", "InterestExpense"),
        ],
        "units": ["USD"],
    },
    "Operating Cash Flow": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        ],
        "units": ["USD"],
    },
    "Capital Expenditures": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
            ("us-gaap", "CapitalExpendituresIncurredButNotYetPaid"),
            ("us-gaap", "PaymentsToAcquireProductiveAssets"),
        ],
        "units": ["USD"],
    },
    "Depreciation & Amortization": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "DepreciationDepletionAndAmortization"),
            ("us-gaap", "DepreciationAmortizationAndAccretionNet"),
            ("us-gaap", "DepreciationAndAmortization"),
            ("us-gaap", "Depreciation"),
        ],
        "units": ["USD"],
    },
    "Diluted Shares": {
        "kind": "duration",
        "concepts": [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")],
        "units": ["shares"],
    },
    "Shares Outstanding": {
        "kind": "instant",
        "concepts": [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ],
        "units": ["shares"],
    },
    "Dividends Paid Common": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "PaymentsOfDividendsCommonStock"),
            ("us-gaap", "PaymentsOfDividends"),
        ],
        "units": ["USD"],
    },
    "Dividends Per Share": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "CommonStockDividendsPerShareDeclared"),
            ("us-gaap", "CommonStockDividendsPerShareCashPaid"),
        ],
        "units": ["USD/shares"],
    },
    "Share Repurchases": {
        "kind": "duration",
        "concepts": [
            ("us-gaap", "PaymentsForRepurchaseOfCommonStock"),
            ("us-gaap", "StockRepurchasedDuringPeriodValue"),
        ],
        "units": ["USD"],
    },
}

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


def _coerce_numeric(value):
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric_value):
        return None
    return numeric_value


def _normalize_sec_fact_entry(entry):
    end_date = entry.get("end")
    value = _coerce_numeric(entry.get("val"))
    if not end_date or value is None:
        return None

    start_date = entry.get("start")
    duration_days = None
    if start_date and end_date:
        try:
            duration_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        except Exception:
            duration_days = None

    return {
        "start": start_date,
        "end": end_date,
        "value": value,
        "filed": str(entry.get("filed") or ""),
        "form": str(entry.get("form") or "").upper(),
        "fp": str(entry.get("fp") or "").upper(),
        "duration_days": duration_days,
    }


def _latest_records_by_key(entries, key_builder):
    latest_entries = {}
    for entry in entries:
        key = key_builder(entry)
        existing = latest_entries.get(key)
        if existing is None or entry["filed"] > existing["filed"]:
            latest_entries[key] = entry
    return latest_entries


def _collect_companyfact_entries(companyfacts_payload, namespace, concept, preferred_units):
    concept_payload = (((companyfacts_payload.get("facts") or {}).get(namespace) or {}).get(concept) or {})
    units_payload = concept_payload.get("units") or {}
    if not units_payload:
        return []

    raw_entries = []
    for unit_name in preferred_units:
        raw_entries.extend(units_payload.get(unit_name, []))

    if not raw_entries:
        first_unit = next(iter(units_payload))
        raw_entries = units_payload.get(first_unit, [])

    normalized_entries = []
    for raw_entry in raw_entries:
        normalized_entry = _normalize_sec_fact_entry(raw_entry)
        if normalized_entry is None:
            continue
        if not (
            normalized_entry["form"].startswith("10-K")
            or normalized_entry["form"].startswith("10-Q")
        ):
            continue
        normalized_entries.append(normalized_entry)

    return normalized_entries


def _annual_duration_record_map(entries):
    annual_entries = [
        entry
        for entry in entries
        if entry["start"]
        and entry["duration_days"] is not None
        and entry["duration_days"] >= 300
        and entry["form"].startswith("10-K")
    ]
    return {
        end_date: entry
        for end_date, entry in _latest_records_by_key(
            annual_entries,
            lambda item: item["end"],
        ).items()
    }


def _quarterly_duration_record_map(entries):
    direct_entries = [
        entry
        for entry in entries
        if entry["start"]
        and entry["duration_days"] is not None
        and 70 <= entry["duration_days"] <= 110
    ]
    cumulative_entries = [
        entry
        for entry in entries
        if entry["start"]
        and entry["duration_days"] is not None
        and entry["duration_days"] > 110
    ]

    direct_entries = list(
        _latest_records_by_key(
            direct_entries,
            lambda item: (item["start"], item["end"], item["duration_days"]),
        ).values()
    )
    cumulative_entries = list(
        _latest_records_by_key(
            cumulative_entries,
            lambda item: (item["start"], item["end"], item["duration_days"]),
        ).values()
    )

    grouped_direct_entries = {}
    for entry in direct_entries:
        grouped_direct_entries.setdefault(entry["start"], {})[entry["end"]] = entry

    grouped_cumulative_entries = {}
    for entry in cumulative_entries:
        grouped_cumulative_entries.setdefault(entry["start"], {})[entry["end"]] = entry

    quarterly_record_map = {}
    for fiscal_start in sorted(set(grouped_direct_entries) | set(grouped_cumulative_entries)):
        direct_end_map = grouped_direct_entries.get(fiscal_start, {})
        cumulative_end_map = grouped_cumulative_entries.get(fiscal_start, {})
        previous_cumulative_value = None

        for end_date in sorted(set(direct_end_map) | set(cumulative_end_map)):
            direct_entry = direct_end_map.get(end_date)
            cumulative_entry = cumulative_end_map.get(end_date)

            if direct_entry is not None:
                quarterly_record_map[end_date] = direct_entry.copy()
            elif cumulative_entry is not None:
                derived_entry = cumulative_entry.copy()
                derived_entry["value"] = (
                    cumulative_entry["value"]
                    if previous_cumulative_value is None
                    else cumulative_entry["value"] - previous_cumulative_value
                )
                derived_entry["derived"] = True
                quarterly_record_map[end_date] = derived_entry

            if cumulative_entry is not None:
                previous_cumulative_value = cumulative_entry["value"]
            elif direct_entry is not None:
                previous_cumulative_value = (
                    direct_entry["value"]
                    if previous_cumulative_value is None
                    else previous_cumulative_value + direct_entry["value"]
                )

    return quarterly_record_map


def _instant_record_map(entries, frequency):
    if frequency == "annual":
        instant_entries = [entry for entry in entries if entry["form"].startswith("10-K")]
    else:
        instant_entries = [
            entry
            for entry in entries
            if entry["form"].startswith("10-Q") or entry["form"].startswith("10-K")
        ]

    return {
        end_date: entry
        for end_date, entry in _latest_records_by_key(
            instant_entries,
            lambda item: item["end"],
        ).items()
    }


def _extract_sec_field_record_map(companyfacts_payload, field_name, frequency):
    field_spec = SEC_FACT_SPECS[field_name]
    merged_record_map = {}

    for namespace, concept in field_spec["concepts"]:
        concept_entries = _collect_companyfact_entries(
            companyfacts_payload,
            namespace,
            concept,
            field_spec["units"],
        )
        if not concept_entries:
            continue

        if field_spec["kind"] == "duration":
            concept_record_map = (
                _quarterly_duration_record_map(concept_entries)
                if frequency == "quarterly"
                else _annual_duration_record_map(concept_entries)
            )
        else:
            concept_record_map = _instant_record_map(concept_entries, frequency)

        for end_date, record in concept_record_map.items():
            merged_record_map.setdefault(end_date, record)

    return merged_record_map


def _build_sec_raw_period_frame(companyfacts_payload, frequency):
    period_rows = {}

    for field_name in SEC_FACT_SPECS:
        field_record_map = _extract_sec_field_record_map(companyfacts_payload, field_name, frequency)
        for end_date, record in field_record_map.items():
            if field_name == "Shares Outstanding" and end_date not in period_rows:
                continue
            row = period_rows.setdefault(
                end_date,
                {
                    "Period End Date": pd.to_datetime(end_date),
                    "_Period Start Date": record.get("start"),
                    "_Duration Days": record.get("duration_days"),
                    "_Filed Date": record.get("filed"),
                },
            )
            row[field_name] = record["value"]
            if record.get("start") and not row.get("_Period Start Date"):
                row["_Period Start Date"] = record["start"]
            if record.get("duration_days") and not row.get("_Duration Days"):
                row["_Duration Days"] = record["duration_days"]
            if record.get("filed") and record["filed"] > str(row.get("_Filed Date") or ""):
                row["_Filed Date"] = record["filed"]

    if not period_rows:
        return pd.DataFrame()

    raw_df = pd.DataFrame(period_rows.values()).sort_values("Period End Date").reset_index(drop=True)
    for column_name in RAW_FACT_COLUMNS:
        if column_name not in raw_df.columns:
            raw_df[column_name] = pd.NA

    for numeric_column in [
        "Capital Expenditures",
        "Dividends Paid Common",
        "Share Repurchases",
        "Interest Expense",
        "Current Debt",
        "Long-term Debt",
    ]:
        raw_df[numeric_column] = pd.to_numeric(raw_df[numeric_column], errors="coerce").abs()

    gross_profit_missing = raw_df["Gross Profit"].isna() & raw_df["Total Revenue"].notna() & raw_df["Cost of Revenue"].notna()
    raw_df.loc[gross_profit_missing, "Gross Profit"] = (
        pd.to_numeric(raw_df.loc[gross_profit_missing, "Total Revenue"], errors="coerce")
        - pd.to_numeric(raw_df.loc[gross_profit_missing, "Cost of Revenue"], errors="coerce")
    )

    cogs_missing = raw_df["Cost of Revenue"].isna() & raw_df["Total Revenue"].notna() & raw_df["Gross Profit"].notna()
    raw_df.loc[cogs_missing, "Cost of Revenue"] = (
        pd.to_numeric(raw_df.loc[cogs_missing, "Total Revenue"], errors="coerce")
        - pd.to_numeric(raw_df.loc[cogs_missing, "Gross Profit"], errors="coerce")
    )

    raw_df["Total Debt"] = (
        pd.to_numeric(raw_df["Current Debt"], errors="coerce").fillna(0)
        + pd.to_numeric(raw_df["Long-term Debt"], errors="coerce").fillna(0)
    )
    raw_df["Depreciation & Amortization"] = pd.to_numeric(
        raw_df["Depreciation & Amortization"],
        errors="coerce",
    )
    raw_df["EBITDA"] = (
        pd.to_numeric(raw_df["Operating Income"], errors="coerce")
        + raw_df["Depreciation & Amortization"].fillna(0)
    )
    raw_df["Free Cash Flow"] = (
        pd.to_numeric(raw_df["Operating Cash Flow"], errors="coerce")
        - pd.to_numeric(raw_df["Capital Expenditures"], errors="coerce")
    )

    return raw_df


def _as_numeric_series(values):
    return pd.to_numeric(values, errors="coerce")


def _safe_divide_series(numerator, denominator):
    numerator_series = _as_numeric_series(numerator)
    denominator_series = _as_numeric_series(denominator)
    return numerator_series / denominator_series.mask(denominator_series == 0)


def _growth_rate(series, periods):
    numeric_series = _as_numeric_series(series)
    previous_series = numeric_series.shift(periods)
    return (numeric_series - previous_series) / previous_series.mask(previous_series == 0)


def _cagr_rate(series, periods, years):
    numeric_series = _as_numeric_series(series)
    previous_series = numeric_series.shift(periods)
    ratio_series = numeric_series / previous_series.mask(previous_series <= 0)
    cagr_series = ratio_series.pow(1 / years) - 1
    cagr_series[(numeric_series <= 0) | (previous_series <= 0)] = pd.NA
    return cagr_series


def _average_balance(series):
    numeric_series = _as_numeric_series(series)
    average_series = (numeric_series + numeric_series.shift(1)) / 2
    return average_series.fillna(numeric_series)


def _latest_non_null_from_frame(df, column_names):
    if df.empty:
        return None

    descending_df = df.sort_values("Period End Date", ascending=False)
    for column_name in column_names:
        if column_name not in descending_df.columns:
            continue
        values = pd.to_numeric(descending_df[column_name], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return None


def _get_current_market_price(ticker):
    stock = yf.Ticker(ticker.upper())

    try:
        fast_info = stock.fast_info
        current_price = _coerce_numeric(fast_info.get("lastPrice"))
        if current_price is not None:
            return current_price
    except Exception:
        pass

    try:
        history = stock.history(period="5d", interval="1d", auto_adjust=False)
        if history is not None and not history.empty:
            return _coerce_numeric(history["Close"].dropna().iloc[-1])
    except Exception:
        pass

    return None


def _build_current_market_snapshot(ticker, companyfacts_payload):
    current_price = _get_current_market_price(ticker)
    quarterly_raw = _build_sec_raw_period_frame(companyfacts_payload, "quarterly")
    annual_raw = _build_sec_raw_period_frame(companyfacts_payload, "annual")
    market_source_df = quarterly_raw if not quarterly_raw.empty else annual_raw

    shares_outstanding = None
    share_record_map = _extract_sec_field_record_map(companyfacts_payload, "Shares Outstanding", "quarterly")
    if not share_record_map:
        share_record_map = _extract_sec_field_record_map(companyfacts_payload, "Shares Outstanding", "annual")
    if share_record_map:
        latest_share_end = max(share_record_map, key=lambda end_date: pd.to_datetime(end_date))
        shares_outstanding = _coerce_numeric(share_record_map[latest_share_end].get("value"))
    if shares_outstanding is None:
        shares_outstanding = _latest_non_null_from_frame(market_source_df, ["Diluted Shares"])
    total_debt = _latest_non_null_from_frame(market_source_df, ["Total Debt", "Long-term Debt"])
    cash_and_equivalents = _latest_non_null_from_frame(market_source_df, ["Cash and Equivalents"])

    current_market_cap = None
    if current_price is not None and shares_outstanding is not None:
        current_market_cap = current_price * shares_outstanding

    current_enterprise_value = None
    if current_market_cap is not None:
        current_enterprise_value = (
            current_market_cap
            + (total_debt or 0.0)
            - (cash_and_equivalents or 0.0)
        )

    return {
        "current_price": current_price,
        "shares_outstanding": shares_outstanding,
        "total_debt": total_debt,
        "cash_and_equivalents": cash_and_equivalents,
        "current_market_cap": current_market_cap,
        "current_enterprise_value": current_enterprise_value,
    }


def _compute_indicator_frame(raw_df, frequency, market_snapshot):
    if raw_df.empty:
        return raw_df

    df = raw_df.copy().sort_values("Period End Date").reset_index(drop=True)
    days_in_period = _as_numeric_series(df["_Duration Days"]).fillna(91 if frequency == "quarterly" else 365)

    revenue = _as_numeric_series(df["Total Revenue"])
    cost_of_revenue = _as_numeric_series(df["Cost of Revenue"])
    gross_profit = _as_numeric_series(df["Gross Profit"])
    operating_income = _as_numeric_series(df["Operating Income"])
    ebitda = _as_numeric_series(df["EBITDA"])
    net_income = _as_numeric_series(df["Net Income"])
    pretax_income = _as_numeric_series(df["Income Before Tax"])
    tax_expense = _as_numeric_series(df["Income Tax Expense"])
    diluted_eps = _as_numeric_series(df["Diluted EPS"])
    total_assets = _as_numeric_series(df["Total Assets"])
    equity = _as_numeric_series(df["Shareholders Equity"])
    current_assets = _as_numeric_series(df["Current Assets"])
    current_liabilities = _as_numeric_series(df["Current Liabilities"])
    cash_and_equivalents = _as_numeric_series(df["Cash and Equivalents"])
    inventory = _as_numeric_series(df["Inventory"])
    receivables = _as_numeric_series(df["Accounts Receivable"])
    payables = _as_numeric_series(df["Accounts Payable"])
    ppe_net = _as_numeric_series(df["PPE Net"])
    current_debt = _as_numeric_series(df["Current Debt"])
    long_term_debt = _as_numeric_series(df["Long-term Debt"])
    total_debt = _as_numeric_series(df["Total Debt"])
    interest_expense = _as_numeric_series(df["Interest Expense"]).abs()
    operating_cash_flow = _as_numeric_series(df["Operating Cash Flow"])
    capex = _as_numeric_series(df["Capital Expenditures"]).abs()
    free_cash_flow = _as_numeric_series(df["Free Cash Flow"])
    depreciation_and_amortization = _as_numeric_series(df["Depreciation & Amortization"])
    dividends_paid = _as_numeric_series(df["Dividends Paid Common"]).abs()
    dividends_per_share = _as_numeric_series(df["Dividends Per Share"])
    share_repurchases = _as_numeric_series(df["Share Repurchases"]).abs()

    current_price = market_snapshot.get("current_price")
    current_market_cap = market_snapshot.get("current_market_cap")
    current_enterprise_value = market_snapshot.get("current_enterprise_value")

    if frequency == "quarterly":
        revenue_basis = revenue.rolling(4, min_periods=4).sum()
        cost_basis = cost_of_revenue.rolling(4, min_periods=4).sum()
        operating_income_basis = operating_income.rolling(4, min_periods=4).sum()
        ebitda_basis = ebitda.rolling(4, min_periods=4).sum()
        net_income_basis = net_income.rolling(4, min_periods=4).sum()
        pretax_income_basis = pretax_income.rolling(4, min_periods=4).sum()
        tax_expense_basis = tax_expense.rolling(4, min_periods=4).sum()
        operating_cash_flow_basis = operating_cash_flow.rolling(4, min_periods=4).sum()
        free_cash_flow_basis = free_cash_flow.rolling(4, min_periods=4).sum()
        interest_expense_basis = interest_expense.rolling(4, min_periods=4).sum()
        dividends_paid_basis = dividends_paid.rolling(4, min_periods=4).sum()
        dividends_per_share_basis = dividends_per_share.rolling(4, min_periods=4).sum()
        share_repurchases_basis = share_repurchases.rolling(4, min_periods=4).sum()
        capex_basis = capex.rolling(4, min_periods=4).sum()
        depreciation_basis = depreciation_and_amortization.rolling(4, min_periods=4).sum()
        eps_basis = diluted_eps.rolling(4, min_periods=4).sum()
        revenue_growth_yoy = _growth_rate(revenue, 4)
        quarterly_revenue_growth = _growth_rate(revenue, 1)
        revenue_cagr = _cagr_rate(revenue, 12, 3)
        eps_growth_yoy = _growth_rate(diluted_eps, 4)
        eps_cagr = _cagr_rate(diluted_eps, 12, 3)
        free_cash_flow_growth = _growth_rate(free_cash_flow, 4)
    else:
        revenue_basis = revenue
        cost_basis = cost_of_revenue
        operating_income_basis = operating_income
        ebitda_basis = ebitda
        net_income_basis = net_income
        pretax_income_basis = pretax_income
        tax_expense_basis = tax_expense
        operating_cash_flow_basis = operating_cash_flow
        free_cash_flow_basis = free_cash_flow
        interest_expense_basis = interest_expense
        dividends_paid_basis = dividends_paid
        dividends_per_share_basis = dividends_per_share
        share_repurchases_basis = share_repurchases
        capex_basis = capex
        depreciation_basis = depreciation_and_amortization
        eps_basis = diluted_eps
        revenue_growth_yoy = _growth_rate(revenue, 1)
        quarterly_revenue_growth = pd.Series(pd.NA, index=df.index, dtype="object")
        revenue_cagr = _cagr_rate(revenue, 3, 3)
        eps_growth_yoy = _growth_rate(diluted_eps, 1)
        eps_cagr = _cagr_rate(diluted_eps, 3, 3)
        free_cash_flow_growth = _growth_rate(free_cash_flow, 1)

    average_assets = _average_balance(total_assets)
    average_equity = _average_balance(equity)
    average_inventory = _average_balance(inventory)
    average_receivables = _average_balance(receivables)
    average_payables = _average_balance(payables)
    average_ppe = _average_balance(ppe_net)
    average_invested_capital = _average_balance(total_debt.fillna(0) + equity.fillna(0) - cash_and_equivalents.fillna(0))

    df["Current Market Price"] = current_price
    df["Current Market Capitalization"] = current_market_cap
    df["Current Enterprise Value"] = current_enterprise_value
    df["Gross Margin"] = _safe_divide_series(gross_profit, revenue)
    df["Operating Margin"] = _safe_divide_series(operating_income, revenue)
    df["Net Profit Margin"] = _safe_divide_series(net_income, revenue)
    df["EBITDA Margin"] = _safe_divide_series(ebitda, revenue)
    df["Return on Equity (ROE)"] = _safe_divide_series(net_income_basis, average_equity)
    df["Return on Assets (ROA)"] = _safe_divide_series(net_income_basis, average_assets)

    effective_tax_rate = _safe_divide_series(tax_expense_basis, pretax_income_basis).clip(lower=0, upper=0.35).fillna(0.21)
    nopat = operating_income_basis * (1 - effective_tax_rate)
    df["Return on Invested Capital (ROIC)"] = _safe_divide_series(nopat, average_invested_capital)
    df["Revenue Growth YoY"] = revenue_growth_yoy
    df["Quarterly Revenue Growth"] = quarterly_revenue_growth
    df["Revenue CAGR 3-Year"] = revenue_cagr
    df["EPS Growth YoY"] = eps_growth_yoy
    df["EPS CAGR 3-Year"] = eps_cagr
    df["Free Cash Flow Growth"] = free_cash_flow_growth
    df["Organic Revenue Growth"] = pd.Series(float("nan"), index=df.index, dtype="float64")

    trailing_pe = pd.Series(float("nan"), index=df.index, dtype="float64")
    if current_price is not None:
        trailing_pe = current_price / eps_basis.mask(eps_basis <= 0)

    forward_eps_proxy = eps_basis * (1 + eps_growth_yoy.clip(lower=-0.95))
    forward_eps_proxy[(eps_basis <= 0) | eps_growth_yoy.isna()] = pd.NA
    pe_forward = pd.Series(float("nan"), index=df.index, dtype="float64")
    if current_price is not None:
        pe_forward = current_price / forward_eps_proxy.mask(forward_eps_proxy <= 0)

    df["Price-to-Earnings (P/E) Trailing"] = trailing_pe
    df["Price-to-Earnings (P/E) Forward"] = pe_forward
    df["PEG Ratio"] = trailing_pe / (eps_growth_yoy * 100).mask((eps_growth_yoy <= 0) | eps_growth_yoy.isna())
    df["Price-to-Book (P/B)"] = (
        pd.Series(current_market_cap, index=df.index, dtype="float64") / equity.mask(equity <= 0)
        if current_market_cap is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["Price-to-Sales (P/S)"] = (
        pd.Series(current_market_cap, index=df.index, dtype="float64") / revenue_basis.mask(revenue_basis <= 0)
        if current_market_cap is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["EV/EBITDA"] = (
        pd.Series(current_enterprise_value, index=df.index, dtype="float64") / ebitda_basis.mask(ebitda_basis <= 0)
        if current_enterprise_value is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["EV/Revenue"] = (
        pd.Series(current_enterprise_value, index=df.index, dtype="float64") / revenue_basis.mask(revenue_basis <= 0)
        if current_enterprise_value is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["Price-to-Free Cash Flow (P/FCF)"] = (
        pd.Series(current_market_cap, index=df.index, dtype="float64") / free_cash_flow_basis.mask(free_cash_flow_basis <= 0)
        if current_market_cap is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )

    df["Current Ratio"] = _safe_divide_series(current_assets, current_liabilities)
    df["Quick Ratio"] = _safe_divide_series(current_assets - inventory.fillna(0), current_liabilities)
    df["Cash Ratio"] = _safe_divide_series(cash_and_equivalents, current_liabilities)
    df["Working Capital Ratio"] = df["Current Ratio"]
    df["Debt-to-Equity (D/E)"] = _safe_divide_series(total_debt, equity)

    net_debt_value = None
    if market_snapshot.get("total_debt") is not None:
        net_debt_value = (market_snapshot.get("total_debt") or 0.0) - (market_snapshot.get("cash_and_equivalents") or 0.0)
    df["Net Debt/EBITDA"] = (
        pd.Series(net_debt_value, index=df.index, dtype="float64") / ebitda_basis.mask(ebitda_basis <= 0)
        if net_debt_value is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["Interest Coverage Ratio"] = _safe_divide_series(operating_income_basis, interest_expense_basis.abs())
    df["Debt Service Coverage Ratio"] = _safe_divide_series(
        operating_cash_flow_basis,
        interest_expense_basis.abs() + current_debt.fillna(0),
    )
    df["Long-term Debt to Capital"] = _safe_divide_series(long_term_debt, long_term_debt + equity)
    df["Asset Turnover"] = _safe_divide_series(revenue_basis, average_assets)
    df["Inventory Turnover"] = _safe_divide_series(cost_basis, average_inventory)
    df["Receivables Turnover"] = _safe_divide_series(revenue_basis, average_receivables)
    df["Payables Turnover"] = _safe_divide_series(cost_basis, average_payables)
    df["Cash Conversion Cycle"] = (
        _safe_divide_series(average_inventory, cost_basis) * days_in_period
        + _safe_divide_series(average_receivables, revenue_basis) * days_in_period
        - _safe_divide_series(average_payables, cost_basis) * days_in_period
    )
    df["Operating Cycle"] = (
        _safe_divide_series(average_inventory, cost_basis) * days_in_period
        + _safe_divide_series(average_receivables, revenue_basis) * days_in_period
    )
    df["Fixed Asset Turnover"] = _safe_divide_series(revenue_basis, average_ppe)
    df["Free Cash Flow Yield"] = (
        free_cash_flow_basis / current_market_cap
        if current_market_cap is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )
    df["FCF / Net Income"] = _safe_divide_series(free_cash_flow_basis, net_income_basis)
    df["CapEx / Depreciation"] = _safe_divide_series(capex_basis, depreciation_basis)
    df["Share Repurchase Yield"] = (
        share_repurchases_basis / current_market_cap
        if current_market_cap is not None
        else pd.Series(float("nan"), index=df.index, dtype="float64")
    )

    dividend_payout_ratio = _safe_divide_series(dividends_paid_basis, net_income_basis)
    dividend_payout_ratio = dividend_payout_ratio.fillna(_safe_divide_series(dividends_per_share_basis, eps_basis))
    df["Dividend Payout Ratio"] = dividend_payout_ratio

    if current_price is not None:
        df["Dividend Yield"] = dividends_per_share_basis / current_price
    elif current_market_cap is not None:
        df["Dividend Yield"] = dividends_paid_basis / current_market_cap
    else:
        df["Dividend Yield"] = pd.Series(float("nan"), index=df.index, dtype="float64")

    df["Reinvestment Rate"] = (1 - dividend_payout_ratio).where(
        dividend_payout_ratio.notna(),
        _safe_divide_series(capex_basis, operating_cash_flow_basis),
    )
    df["Operating Cash Flow Margin"] = _safe_divide_series(operating_cash_flow, revenue)
    df["Free Cash Flow Margin"] = _safe_divide_series(free_cash_flow, revenue)
    df["Cash Flow Return on Investment (CFROI)"] = _safe_divide_series(
        operating_cash_flow_basis,
        average_invested_capital,
    )

    ordered_columns = [column for column in FINANCIAL_INDICATOR_COLUMN_ORDER if column in df.columns]
    return df[ordered_columns + [column for column in df.columns if column not in ordered_columns]]


def _load_sec_companyfacts_payload(ticker):
    ticker = ticker.upper()
    ticker_map = _load_sec_ticker_map()
    sec_identity = ticker_map.get(ticker)
    if not sec_identity:
        raise ValueError(f"Ticker {ticker} was not found in the SEC ticker map.")
    return _sec_get_json(SEC_COMPANYFACTS_URL_TEMPLATE.format(cik=sec_identity["cik"]))


def get_historical_financial_indicators(
    ticker,
    frequency="quarterly",
    max_periods=None,
    companyfacts_payload=None,
    market_snapshot=None,
):
    ticker = ticker.upper()
    frequency = frequency.lower()
    freq_label = "Quarterly" if frequency == "quarterly" else "Annual"

    companyfacts_payload = companyfacts_payload or _load_sec_companyfacts_payload(ticker)
    market_snapshot = market_snapshot or _build_current_market_snapshot(ticker, companyfacts_payload)

    raw_df = _build_sec_raw_period_frame(companyfacts_payload, frequency)
    if raw_df.empty:
        print(f"{ticker}: No SEC Company Facts {freq_label} data available")
        return pd.DataFrame()

    indicator_df = _compute_indicator_frame(raw_df, frequency, market_snapshot)
    indicator_df = indicator_df.sort_values("Period End Date", ascending=False).reset_index(drop=True)
    if max_periods is not None:
        indicator_df = indicator_df.head(max_periods).copy()

    indicator_df["Ticker"] = ticker
    indicator_df["Frequency"] = freq_label
    indicator_df["Period End Date"] = pd.to_datetime(indicator_df["Period End Date"]).dt.strftime("%Y-%m-%d")

    ordered_columns = [column for column in FINANCIAL_INDICATOR_COLUMN_ORDER if column in indicator_df.columns]
    return indicator_df[ordered_columns + [column for column in indicator_df.columns if column not in ordered_columns]]


def _quote_sql_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def _ensure_financial_indicators_table(conn, df):
    existing_columns = [
        row[1]
        for row in conn.execute("PRAGMA table_info(financial_indicators)").fetchall()
    ]

    if not existing_columns:
        column_definitions = []
        for column_name in df.columns:
            column_type = "TEXT" if column_name in IDENTIFIER_COLUMNS else "FLOAT"
            column_definitions.append(f"{_quote_sql_identifier(column_name)} {column_type}")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS financial_indicators ({', '.join(column_definitions)})"
        )
        return

    for column_name in df.columns:
        if column_name in existing_columns:
            continue
        column_type = "TEXT" if column_name in IDENTIFIER_COLUMNS else "FLOAT"
        conn.execute(
            f"ALTER TABLE financial_indicators "
            f"ADD COLUMN {_quote_sql_identifier(column_name)} {column_type}"
        )


def initiate_sql_table(db_path=DEFAULT_STOCK_DB_PATH):
    check = input("it deletes old data in stock_data file...run with caution, are you sure you want to delete and init? Y/N")
    if check.upper() == "Y":
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS financial_indicators")
        _ensure_financial_indicators_table(conn, pd.DataFrame(columns=FINANCIAL_INDICATOR_COLUMN_ORDER))
        conn.commit()
        conn.close()
        print("Table dropped and recreated successfully.")
    else:
        print("sql table not initialized...")
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

        ordered_columns = [column for column in FINANCIAL_INDICATOR_COLUMN_ORDER if column in df.columns]
        df = df[ordered_columns + [column for column in df.columns if column not in ordered_columns]]
        md_table = df.to_markdown(index=False)

        text = (
            f"**Latest {len(df)} {freq} Financial Indicators - {ticker}**\n\n"
            f"Company: {company_profile['company_name']}\n"
            f"Sector: {company_profile['sector'] or 'Unknown'}\n"
            f"Industry: {company_profile['industry'] or 'Unknown'}\n\n"
            f"Current market price used for market-based ratios: {df['Current Market Price'].iloc[0] if 'Current Market Price' in df.columns else 'Unavailable'}\n"
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
                "source": "SEC Company Facts + current market price + SQLite",
                "table_rows": len(df),
                "ten_k_count": len(company_profile.get("ten_k_filings", [])),
                "ten_q_count": len(company_profile.get("ten_q_filings", [])),
            }
        )
        documents.append(doc)

    conn.close()
    return documents

def update_financial_records(ticker, db_path=DEFAULT_STOCK_DB_PATH):
    conn = sqlite3.connect(db_path)
    ticker = ticker.upper()
    companyfacts_payload = _load_sec_companyfacts_payload(ticker)
    market_snapshot = _build_current_market_snapshot(ticker, companyfacts_payload)

    for freq, max_keep in [('annual', 10), ('quarterly', 12)]:
        freq_label = freq.capitalize()

        df_new = get_historical_financial_indicators(
            ticker,
            frequency=freq,
            max_periods=None,
            companyfacts_payload=companyfacts_payload,
            market_snapshot=market_snapshot,
        )

        if df_new.empty:
            print(f"Skipping update for {ticker} {freq_label}: No new SEC-derived data available")
            continue

        _ensure_financial_indicators_table(conn, df_new)

        df_trimmed = df_new.copy()
        df_trimmed['Period End Date'] = pd.to_datetime(df_trimmed['Period End Date'], errors='coerce')
        df_trimmed = (
            df_trimmed.dropna(subset=['Period End Date'])
            .sort_values('Period End Date', ascending=False)
            .reset_index(drop=True)
            .head(max_keep)
        )

        delete_query = """
        DELETE FROM financial_indicators
        WHERE Ticker = ? AND Frequency = ?
        """
        conn.execute(delete_query, (ticker, freq_label))
        _ensure_financial_indicators_table(conn, df_trimmed)
        df_to_store = df_trimmed.copy()
        df_to_store['Period End Date'] = df_to_store['Period End Date'].dt.strftime('%Y-%m-%d')
        ordered_columns = [column for column in FINANCIAL_INDICATOR_COLUMN_ORDER if column in df_to_store.columns]
        df_to_store = df_to_store[ordered_columns + [column for column in df_to_store.columns if column not in ordered_columns]]
        df_to_store.to_sql('financial_indicators', conn, if_exists='append', index=False)

        print(
            f"Updated {ticker} {freq_label}: Kept {len(df_to_store)} periods "
            f"(latest: {df_to_store['Period End Date'].iloc[0]})"
        )

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

    import ingest_graph
    import ingest_macro

    ingest_graph.refresh_full_graph_for_ticker(
        ticker,
        stock_docs=docs,
        stock_db_path=db_path,
        macro_db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
        filings_base_dir=filings_base_dir,
    )

    print(f"Index successfully refreshed for {ticker}")


#refresh_ticker_data_and_index('aapl')
