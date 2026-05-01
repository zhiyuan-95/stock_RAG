import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import statistics
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
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_EMBED_MODEL = "voyage-finance-2"
DEFAULT_EMBED_DIMENSION = 1024
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ARCHIVES_DOCUMENT_URL_TEMPLATE = (
        "https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_no_dashes}/{primary_document}"
    )
DEFAULT_SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
_SEC_TICKER_MAP = None

INDUSTRY_SECTOR_TAXONOMY = {
    "Energy": ["Energy"],
    "Materials": ["Materials"],
    "Industrials": [
        "Capital Goods",
        "Commercial & Professional Services",
        "Transportation",
    ],
    "Consumer Discretionary": [
        "Automobiles & Components",
        "Consumer Durables & Apparel",
        "Consumer Services",
        "E-Commerce & Direct-to-Consumer",
        "Brick-and-Mortar & Specialty Retail",
    ],
    "Consumer Staples": [
        "Consumer Staples Distribution & Retail",
        "Food, Beverage & Tobacco",
        "Household & Personal Products",
    ],
    "Health Care": [
        "Health Care Equipment & Services",
        "Pharmaceuticals, Biotechnology & Life Sciences",
    ],
    "Financials": [
        "Banks",
        "Financial Services",
        "Insurance",
    ],
    "Information Technology": [
        "Semiconductors & Semiconductor Equipment",
        "Software & Services",
        "Technology Hardware & Equipment",
    ],
    "Communication Services": [
        "Media & Entertainment",
        "Telecommunication Services",
    ],
    "Utilities": ["Utilities"],
    "Real Estate": [
        "Equity Real Estate Investment Trusts (REITs)",
        "Real Estate Management & Development",
    ],
}
SECTOR_TO_INDUSTRY = {
    sector: industry
    for industry, sectors in INDUSTRY_SECTOR_TAXONOMY.items()
    for sector in sectors
}

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
ACTIVE_FULLTEXT_FILING_LIMITS = {
    "10-K": 3,
    "10-Q": 8,
}
FILING_NARRATIVE_SECTIONS = {
    "10-K": [
        ("item_1_business", "SEC 10-K Item 1 Business"),
        ("item_1a_risk_factors", "SEC 10-K Item 1A Risk Factors"),
        ("item_7_mda", "SEC 10-K Item 7 MD&A"),
    ],
    "10-Q": [
        ("item_2_mda", "SEC 10-Q Item 2 MD&A"),
        ("item_1a_risk_factors", "SEC 10-Q Item 1A Risk Factors"),
    ],
}
FILING_STATEMENT_SECTIONS = {
    "10-K": ("item_8_financial_statements", "SEC 10-K Item 8 Financial Statements"),
    "10-Q": ("item_1_financial_statements", "SEC 10-Q Item 1 Financial Statements"),
}
STATEMENT_FACT_COLUMNS = [
    ("Total Revenue", "Revenue"),
    ("Gross Profit", "Gross Profit"),
    ("Operating Income", "Operating Income"),
    ("Net Income", "Net Income"),
    ("Diluted EPS", "Diluted EPS"),
    ("Operating Cash Flow", "Operating Cash Flow"),
    ("Capital Expenditures", "Capital Expenditures"),
    ("Free Cash Flow", "Free Cash Flow"),
    ("Total Debt", "Total Debt"),
    ("Cash and Equivalents", "Cash and Equivalents"),
    ("Shareholders Equity", "Shareholders Equity"),
    ("Shares Outstanding", "Shares Outstanding"),
]
STATEMENT_LINKED_FACT_SPECS = [
    {"label": "Segment revenue", "all": ["segment"], "any": ["revenue", "sales"]},
    {
        "label": "Geographic revenue",
        "any": [
            "geographic",
            "international",
            "americas",
            "emea",
            "europe",
            "greater china",
            "asia pacific",
        ],
    },
    {
        "label": "Debt maturity amounts",
        "any": ["maturit", "due in", "senior notes", "convertible notes", "debt"],
    },
    {"label": "Lease obligations", "any": ["lease", "operating lease", "finance lease"]},
    {"label": "Buybacks", "any": ["repurchase", "buyback", "treasury stock"]},
    {"label": "Impairments", "any": ["impair", "write-down", "write off"]},
    {"label": "Goodwill", "any": ["goodwill"]},
    {"label": "Tax rate", "any": ["effective tax rate", "income tax", "tax rate"]},
    {"label": "Provisions", "any": ["provision", "contingenc", "litigation"]},
    {"label": "Reserve changes", "any": ["reserve", "allowance"]},
]
FINANCIAL_SECTOR_NOTE_SPECS = [
    {"label": "Loan quality / charge-offs", "any": ["charge-off", "nonperform", "delinquen", "credit quality"]},
    {"label": "Deposit mix", "any": ["deposit", "brokered deposits", "noninterest-bearing", "time deposits"]},
    {"label": "Net interest margin drivers", "any": ["net interest margin", "net interest income", "funding costs"]},
    {"label": "Reserve methodology", "any": ["cecl", "allowance", "reserve methodology"]},
    {"label": "AUM / AUA changes", "any": ["assets under management", "aum", "assets under administration", "aua"]},
    {"label": "Fair value hierarchy", "any": ["fair value", "level 1", "level 2", "level 3"]},
    {"label": "Capital ratios", "any": ["cet1", "tier 1", "risk-based capital", "leverage ratio"]},
    {"label": "Underwriting reserves", "any": ["underwriting", "loss reserves", "claim reserves", "policy benefits"]},
]
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

def _require_config_value(name):
    value = os.getenv(name)
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} is not set in config.env or the environment.")
    return normalized


def get_gemini_api_key():
    return _require_config_value("Gemini_API_KEY")


def get_voyage_api_key():
    return _require_config_value("Voyage_API_KEY")


def env(llm_model=None):
    from llama_index.embeddings.voyageai import VoyageEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    load_dotenv("config.env")

    gemini_api_key = get_gemini_api_key()
    voyage_api_key = get_voyage_api_key()
    llm_model = llm_model or os.getenv("STOCK_LLM_MODEL") or DEFAULT_LLM_MODEL

    Settings.llm = GoogleGenAI(
        model=llm_model,
        api_key=gemini_api_key,
        temperature=0.1,
    )
    Settings.embed_model = VoyageEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        voyage_api_key=voyage_api_key,
    )

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

    if code_text.startswith(("48", "49")):
        return "Utilities"
    if any(keyword in description for keyword in ["reit", "real estate investment trust"]):
        return "Equity Real Estate Investment Trusts (REITs)"
    if any(keyword in description for keyword in ["real estate", "property", "land subdivision", "real estate management", "development"]):
        return "Real Estate Management & Development"
    if any(keyword in description for keyword in ["insurance", "insurer", "underwriter", "title insurance"]):
        return "Insurance"
    if any(keyword in description for keyword in ["bank", "savings institution", "national commercial bank", "state commercial bank", "credit union"]):
        return "Banks"
    if any(keyword in description for keyword in ["asset", "investment", "capital", "broker", "securities", "finance", "financial", "consumer lending", "mortgage banker", "exchange"]):
        return "Financial Services"
    if any(keyword in description for keyword in ["pharmaceutical", "biotech", "biological", "life science", "drug", "medicinal"]):
        return "Pharmaceuticals, Biotechnology & Life Sciences"
    if any(keyword in description for keyword in ["medical", "health care", "healthcare", "hospital", "clinic", "laboratory", "dental", "medical equipment", "health services"]):
        return "Health Care Equipment & Services"
    if any(keyword in description for keyword in ["semiconductor", "integrated circuit", "electronic component", "silicon", "chip"]):
        return "Semiconductors & Semiconductor Equipment"
    if any(keyword in description for keyword in ["software", "data processing", "internet", "cloud", "it service", "computer programming", "prepackaged software"]):
        return "Software & Services"
    if any(keyword in description for keyword in ["computer", "communications equipment", "telephone apparatus", "electronic computer", "computer peripheral", "storage device"]):
        return "Technology Hardware & Equipment"
    if any(keyword in description for keyword in ["telecom", "telecommunication", "wireless", "telephone communications", "communications services"]):
        return "Telecommunication Services"
    if any(keyword in description for keyword in ["media", "entertainment", "broadcast", "cable", "publishing", "motion picture", "streaming"]):
        return "Media & Entertainment"
    if any(keyword in description for keyword in ["oil", "gas", "pipeline", "drilling", "exploration", "refining", "coal"]):
        return "Energy"
    if any(keyword in description for keyword in ["chemical", "mining", "metal", "forest product", "paper", "container", "packaging", "glass", "construction materials"]):
        return "Materials"
    if any(keyword in description for keyword in ["railroad", "air freight", "airline", "trucking", "ship", "marine", "logistics", "courier", "transportation"]):
        return "Transportation"
    if any(keyword in description for keyword in ["consulting", "advertising", "employment", "staffing", "waste", "security service", "professional", "business services"]):
        return "Commercial & Professional Services"
    if any(keyword in description for keyword in ["machinery", "aerospace", "defense", "manufacturing", "construction", "electrical equipment", "industrial", "building product"]):
        return "Capital Goods"
    if any(keyword in description for keyword in ["motor vehicles", "passenger car", "automotive", "auto parts", "motorcycles", "truck trailer"]):
        return "Automobiles & Components"
    if any(keyword in description for keyword in ["apparel", "footwear", "textile", "furniture", "home furnishings", "sporting goods", "durable"]):
        return "Consumer Durables & Apparel"
    if any(keyword in description for keyword in ["restaurant", "hotel", "motel", "casino", "lodging", "leisure", "education service"]):
        return "Consumer Services"
    if any(keyword in description for keyword in ["mail-order", "catalog", "direct marketing", "electronic shopping", "e-commerce", "online retail"]):
        return "E-Commerce & Direct-to-Consumer"
    if any(keyword in description for keyword in ["grocery", "drug store", "food store", "wholesale", "warehouse club", "convenience store"]):
        return "Consumer Staples Distribution & Retail"
    if any(keyword in description for keyword in ["beverage", "food", "tobacco", "brew", "distill", "soft drink", "meat packing"]):
        return "Food, Beverage & Tobacco"
    if any(keyword in description for keyword in ["household", "personal", "cosmetic", "toiletries", "soap", "cleaning preparation"]):
        return "Household & Personal Products"
    if any(keyword in description for keyword in ["retail", "department store", "specialty store", "dealer", "shopping"]):
        return "Brick-and-Mortar & Specialty Retail"
    return None


def _derive_industry_from_sic(sic_code, sic_description):
    sector = _derive_sector_from_sic(sic_code, sic_description)
    if not sector:
        return None
    return SECTOR_TO_INDUSTRY.get(sector)


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
        "sector": _clean_profile_field(info.get("industry")),
        "industry": _clean_profile_field(info.get("sector")),
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


def _sorted_filing_records(filing_records):
    return sorted(
        filing_records,
        key=lambda record: pd.to_datetime(record.get("filing_date"), errors="coerce"),
        reverse=True,
    )


def _dedupe_filing_records(filing_records):
    deduped_records = []
    seen_keys = set()

    for filing_record in _sorted_filing_records(filing_records):
        accession_number = str(filing_record.get("accession_number") or "").strip()
        primary_document = str(filing_record.get("primary_document") or "").strip()
        filing_date = str(filing_record.get("filing_date") or "").strip()
        dedupe_key = accession_number or f"{filing_date}|{primary_document}"
        if not dedupe_key or dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_records.append(filing_record)

    return deduped_records


def _roll_filing_window(filing_records, max_records):
    rolled_records = _dedupe_filing_records(filing_records)
    if max_records is None:
        return rolled_records
    return rolled_records[:max_records]


def _fiscal_quarter_from_period(form_type, fiscal_period):
    period = str(fiscal_period or "").upper().strip()
    if period.startswith("Q") and len(period) >= 2:
        return period[:2]
    if form_type == "10-K":
        return "FY"
    return None


def _filing_retrieval_tier(form_type, recency_rank):
    return "active" if recency_rank < ACTIVE_FULLTEXT_FILING_LIMITS.get(form_type, 0) else "archive"


def _company_profile_from_filing_record(filing_record):
    sic_description = filing_record.get("sic_description")
    return {
        "company_name": filing_record.get("company_name") or filing_record.get("ticker"),
        "sector": _derive_sector_from_sic(filing_record.get("sic"), sic_description),
        "industry": _derive_industry_from_sic(filing_record.get("sic"), sic_description),
        "sector_source": "SEC SIC-derived sector" if sic_description else None,
        "industry_source": "SEC SIC-derived industry" if sic_description else None,
        "peer_tickers": [],
        "peer_source": None,
    }


def _finalize_filing_records(ticker, form_type, filing_records):
    finalized_records = []
    for recency_rank, filing_record in enumerate(_sorted_filing_records(filing_records)):
        finalized_record = dict(filing_record)
        finalized_record["form_type"] = form_type
        finalized_record["recency_rank"] = recency_rank
        finalized_record["retrieval_tier"] = _filing_retrieval_tier(form_type, recency_rank)
        if finalized_record.get("sections"):
            finalized_record["documents"] = _document_payloads_from_section_map(
                ticker.upper(),
                _company_profile_from_filing_record(finalized_record),
                finalized_record,
            )
        finalized_records.append(finalized_record)
    return finalized_records


def _filing_base_metadata(
    ticker,
    company_profile,
    filing_record,
    doc_type,
    section_key,
    section_title,
):
    fiscal_period = filing_record.get("fiscal_period")
    return {
        "ticker": ticker,
        "type": doc_type,
        "form_type": filing_record["form_type"],
        "section_key": section_key,
        "section_title": section_title,
        "section_name": section_title,
        "company_name": company_profile["company_name"],
        "sector": company_profile["sector"] or "Unknown",
        "industry": company_profile["industry"] or "Unknown",
        "sector_source": company_profile.get("sector_source"),
        "industry_source": company_profile.get("industry_source"),
        "peer_tickers": company_profile.get("peer_tickers", []),
        "peer_count": len(company_profile.get("peer_tickers", [])),
        "peer_source": company_profile.get("peer_source"),
        "filing_release_date": filing_record.get("filing_release_date") or filing_record["filing_date"],
        "next_release_date": filing_record.get("next_release_date"),
        "next_release_date_source": filing_record.get("next_release_date_source"),
        "fiscal_year_end": filing_record.get("fiscal_year_end") or filing_record.get("reporting_date"),
        "filer_status": filing_record.get("filer_status"),
        "filer_status_label": _humanize_filer_status(filing_record.get("filer_status")),
        "filing_deadline_days": filing_record.get("filing_deadline_days"),
        "filing_lag_days": filing_record.get("filing_lag_days"),
        "historical_filing_lag_days": filing_record.get("historical_filing_lag_days"),
        "filing_date": filing_record["filing_date"],
        "filing_url": filing_record["filing_url"],
        "accession_number": filing_record["accession_number"],
        "reporting_date": filing_record.get("reporting_date"),
        "fiscal_year": filing_record.get("fiscal_year"),
        "fiscal_period": fiscal_period,
        "fiscal_quarter": _fiscal_quarter_from_period(filing_record["form_type"], fiscal_period),
        "retrieval_tier": filing_record.get("retrieval_tier", "archive"),
        "filing_rank": filing_record.get("recency_rank"),
        "source": section_title,
    }


def _section_summary_text(text, max_sentences=3, max_chars=520):
    cleaned_text = _clean_section_text(text)
    if not cleaned_text:
        return ""

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n{2,}", cleaned_text)
        if len(paragraph.strip()) >= 80
    ]
    candidate_text = " ".join(paragraphs[:3]) if paragraphs else cleaned_text
    sentence_candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", candidate_text)

    selected_sentences = []
    for sentence in sentence_candidates:
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        if len(cleaned_sentence) < 40:
            continue
        if cleaned_sentence.lower().startswith("item "):
            continue
        selected_sentences.append(cleaned_sentence)
        if len(selected_sentences) >= max_sentences:
            break

    summary_text = " ".join(selected_sentences) if selected_sentences else candidate_text
    return _truncate_text(summary_text, max_chars=max_chars)


def _narrative_section_payloads(ticker, company_profile, filing_record, section_key, section_title, section_text):
    metadata = _filing_base_metadata(
        ticker,
        company_profile,
        filing_record,
        doc_type="sec_filing_section",
        section_key=section_key,
        section_title=section_title,
    )
    summary_text = _section_summary_text(section_text)

    payloads = [
        {
            "text": (
                f"**{section_title}**\n\n"
                f"Company: {company_profile['company_name']}\n"
                f"Ticker: {ticker}\n"
                f"Form Type: {filing_record['form_type']}\n"
                f"Filing Date: {filing_record['filing_date']}\n"
                f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}\n"
                f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}\n"
                f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}\n"
                f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}\n"
                f"Filing URL: {filing_record['filing_url']}\n"
                f"Industry: {company_profile['industry'] or 'Unknown'}\n"
                f"Sector: {company_profile['sector'] or 'Unknown'}\n\n"
                f"{section_text}"
            ),
            "metadata": metadata,
        }
    ]

    if summary_text:
        summary_metadata = metadata | {
            "type": "sec_section_summary",
            "source": f"{section_title} summary",
            "section_summary": summary_text,
        }
        payloads.append(
            {
                "text": (
                    f"**{section_title} Summary**\n\n"
                    f"Company: {company_profile['company_name']}\n"
                    f"Ticker: {ticker}\n"
                    f"Form Type: {filing_record['form_type']}\n"
                    f"Filing Date: {filing_record['filing_date']}\n"
                    f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}\n"
                    f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}\n"
                    f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}\n"
                    f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}\n\n"
                    f"Short summary:\n{summary_text}"
                ),
                "metadata": summary_metadata,
            }
        )

    return payloads


def _document_payloads_from_section_map(ticker, company_profile, filing_record):
    form_type = filing_record["form_type"]
    section_map = filing_record["sections"]
    payloads = []

    for section_key, section_title in FILING_NARRATIVE_SECTIONS[form_type]:
        section_text = section_map.get(section_key) or ""
        if not section_text:
            continue
        payloads.extend(
            _narrative_section_payloads(
                ticker,
                company_profile,
                filing_record,
                section_key,
                section_title,
                section_text,
            )
        )

    return payloads


def _persist_filing_payloads(
    ticker,
    form_type,
    filing_records,
    filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR,
    max_records=None,
):
    form_dir = _filing_form_dir(ticker, form_type, filings_base_dir=filings_base_dir)
    os.makedirs(form_dir, exist_ok=True)

    filing_records = _roll_filing_window(filing_records, max_records=max_records)
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


def _format_date_like(value):
    parsed_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed_value):
        return None
    return parsed_value.strftime("%Y-%m-%d")


def _infer_next_release_date(form_type, filing_date):
    formatted_filing_date = _format_date_like(filing_date)
    if not formatted_filing_date:
        return None

    filing_timestamp = pd.Timestamp(formatted_filing_date)
    offset = pd.DateOffset(years=1) if form_type == "10-K" else pd.DateOffset(months=3)
    return (filing_timestamp + offset).strftime("%Y-%m-%d")


def _extract_filer_status(filing_text):
    if not filing_text:
        return None

    cover_text = re.sub(r"\s+", " ", filing_text[:25000])
    checkbox_patterns = [
        (r"large accelerated filer\s*[:\-]?\s*[\[\(]?(?:x|✓|✔|☒|þ|yes)\b", "large_accelerated_filer"),
        (r"accelerated filer\s*[:\-]?\s*[\[\(]?(?:x|✓|✔|☒|þ|yes)\b", "accelerated_filer"),
        (r"non-accelerated filer\s*[:\-]?\s*[\[\(]?(?:x|✓|✔|☒|þ|yes)\b", "non_accelerated_filer"),
        (r"smaller reporting company\s*[:\-]?\s*[\[\(]?(?:x|✓|✔|☒|þ|yes)\b", "smaller_reporting_company"),
    ]
    for pattern, filer_status in checkbox_patterns:
        if re.search(pattern, cover_text, flags=re.IGNORECASE):
            return filer_status

    simple_mentions = [
        ("large accelerated filer", "large_accelerated_filer"),
        ("accelerated filer", "accelerated_filer"),
        ("non-accelerated filer", "non_accelerated_filer"),
        ("smaller reporting company", "smaller_reporting_company"),
    ]
    found_statuses = [
        filer_status
        for label, filer_status in simple_mentions
        if label in cover_text.lower()
    ]
    if len(found_statuses) == 1:
        return found_statuses[0]
    return None


def _filer_deadline_days(filer_status):
    return {
        "large_accelerated_filer": 60,
        "accelerated_filer": 75,
        "non_accelerated_filer": 90,
        "smaller_reporting_company": 90,
    }.get(str(filer_status or "").strip().lower())


def _humanize_filer_status(filer_status):
    return {
        "large_accelerated_filer": "Large accelerated filer",
        "accelerated_filer": "Accelerated filer",
        "non_accelerated_filer": "Non-accelerated filer",
        "smaller_reporting_company": "Smaller reporting company",
    }.get(str(filer_status or "").strip().lower())


def _filing_lag_days(filing_record):
    report_date = pd.to_datetime(filing_record.get("reporting_date"), errors="coerce")
    filing_release_date = pd.to_datetime(
        filing_record.get("filing_release_date") or filing_record.get("filing_date"),
        errors="coerce",
    )
    if pd.isna(report_date) or pd.isna(filing_release_date):
        return None

    lag_days = int((filing_release_date.normalize() - report_date.normalize()).days)
    if lag_days < 0:
        return None
    return lag_days


def _estimate_latest_10k_next_release_date(filing_records):
    sorted_records = _sorted_filing_records(filing_records)
    if not sorted_records:
        return None, None, None, None

    latest_record = sorted_records[0]
    fiscal_year_end = _format_date_like(latest_record.get("reporting_date"))
    if not fiscal_year_end:
        return None, None, None, None

    recent_lags = [
        lag_days
        for lag_days in (_filing_lag_days(record) for record in sorted_records[:5])
        if lag_days is not None
    ]
    historical_lag_days = (
        int(round(statistics.median(recent_lags)))
        if recent_lags
        else None
    )

    filer_status = (
        latest_record.get("filer_status")
        or next(
            (
                record.get("filer_status")
                for record in sorted_records
                if record.get("filer_status")
            ),
            None,
        )
    )
    deadline_days = _filer_deadline_days(filer_status)

    next_fiscal_year_end = pd.Timestamp(fiscal_year_end) + pd.DateOffset(years=1)
    if historical_lag_days is not None:
        applied_lag_days = min(historical_lag_days, deadline_days) if deadline_days is not None else historical_lag_days
        next_release_date = (next_fiscal_year_end + pd.Timedelta(days=applied_lag_days)).strftime("%Y-%m-%d")
        return (
            next_release_date,
            "estimated_from_fiscal_year_end_and_historical_10k_pattern",
            historical_lag_days,
            deadline_days,
        )

    if deadline_days is not None:
        next_release_date = (next_fiscal_year_end + pd.Timedelta(days=deadline_days)).strftime("%Y-%m-%d")
        return (
            next_release_date,
            "estimated_from_fiscal_year_end_and_filer_deadline",
            None,
            deadline_days,
        )

    fallback_date = _infer_next_release_date("10-K", latest_record.get("filing_release_date") or latest_record.get("filing_date"))
    return fallback_date, "inferred_from_filing_release_date", None, None


def _infer_latest_10q_fiscal_quarter(latest_record, reference_fiscal_year_end=None):
    explicit_quarter = (
        latest_record.get("fiscal_quarter")
        or _fiscal_quarter_from_period("10-Q", latest_record.get("fiscal_period"))
    )
    if explicit_quarter in {"Q1", "Q2", "Q3"}:
        return explicit_quarter

    quarter_end = pd.to_datetime(latest_record.get("reporting_date"), errors="coerce")
    fiscal_year_end = pd.to_datetime(reference_fiscal_year_end, errors="coerce")
    if pd.isna(quarter_end) or pd.isna(fiscal_year_end):
        return None

    for fiscal_quarter, months_until_year_end in (("Q3", 3), ("Q2", 6), ("Q1", 9)):
        candidate_year_end = quarter_end + pd.DateOffset(months=months_until_year_end)
        if abs((candidate_year_end.normalize() - fiscal_year_end.normalize()).days) <= 5:
            return fiscal_quarter

    return None


def _estimate_latest_10q_next_release_date(
    filing_records,
    reference_filer_status=None,
    reference_fiscal_year_end=None,
):
    sorted_records = _sorted_filing_records(filing_records)
    if not sorted_records:
        return None, None, None

    latest_record = sorted_records[0]
    quarter_end = _format_date_like(latest_record.get("reporting_date"))
    if not quarter_end:
        return None, None, None

    fiscal_quarter = _infer_latest_10q_fiscal_quarter(
        latest_record,
        reference_fiscal_year_end=reference_fiscal_year_end,
    )
    filer_status = (
        latest_record.get("filer_status")
        or reference_filer_status
        or next(
            (
                record.get("filer_status")
                for record in sorted_records
                if record.get("filer_status")
            ),
            None,
        )
    )
    deadline_days = 40 if filer_status in {"large_accelerated_filer", "accelerated_filer"} else 45

    quarter_end_ts = pd.Timestamp(quarter_end)
    month_offset = 6 if fiscal_quarter == "Q3" else 3
    next_quarter_end = quarter_end_ts + pd.DateOffset(months=month_offset)
    next_release_date = (next_quarter_end + pd.Timedelta(days=deadline_days)).strftime("%Y-%m-%d")
    return next_release_date, "estimated_from_quarter_end_and_filer_deadline", deadline_days


def _refresh_filing_schedule_metadata(
    form_type,
    filing_records,
    reference_filer_status=None,
    reference_fiscal_year_end=None,
):
    sorted_records = _sorted_filing_records(
        [_normalize_filing_release_metadata(form_type, filing_record) for filing_record in filing_records])
    if not sorted_records:
        return []

    latest_estimate = (None, None, None, None)
    if form_type == "10-K":
        latest_estimate = _estimate_latest_10k_next_release_date(sorted_records)
    elif form_type == "10-Q":
        latest_estimate = _estimate_latest_10q_next_release_date(
            sorted_records,
            reference_filer_status=reference_filer_status,
            reference_fiscal_year_end=reference_fiscal_year_end)

    refreshed_records = []
    for index, filing_record in enumerate(sorted_records):
        normalized_record = dict(filing_record)
        normalized_record["filing_release_date"] = (
            _format_date_like(normalized_record.get("filing_release_date"))
            or _format_date_like(normalized_record.get("filing_date"))
        )
        normalized_record["fiscal_year_end"] = _format_date_like(normalized_record.get("reporting_date"))
        lag_days = _filing_lag_days(normalized_record)
        if lag_days is not None:
            normalized_record["filing_lag_days"] = lag_days

        if index > 0:
            next_record = sorted_records[index - 1]
            normalized_record["next_release_date"] = (
                _format_date_like(next_record.get("filing_release_date"))
                or _format_date_like(next_record.get("filing_date"))
            )
            normalized_record["next_release_date_source"] = "actual_next_filing_release_date"
        elif form_type == "10-K":
            next_release_date, next_release_source, historical_lag_days, deadline_days = latest_estimate
            normalized_record["next_release_date"] = next_release_date
            normalized_record["next_release_date_source"] = next_release_source
            if historical_lag_days is not None:
                normalized_record["historical_filing_lag_days"] = historical_lag_days
            if deadline_days is not None:
                normalized_record["filing_deadline_days"] = deadline_days
        elif form_type == "10-Q":
            next_release_date, next_release_source, deadline_days = latest_estimate
            normalized_record["next_release_date"] = next_release_date
            normalized_record["next_release_date_source"] = next_release_source
            normalized_record["filer_status"] = (
                normalized_record.get("filer_status") or reference_filer_status
            )
            inferred_quarter = _infer_latest_10q_fiscal_quarter(
                normalized_record,
                reference_fiscal_year_end=reference_fiscal_year_end,
            )
            if inferred_quarter and not normalized_record.get("fiscal_quarter"):
                normalized_record["fiscal_quarter"] = inferred_quarter
            if deadline_days is not None:
                normalized_record["filing_deadline_days"] = deadline_days
        else:
            normalized_record["next_release_date"] = (
                _format_date_like(normalized_record.get("next_release_date"))
                or _infer_next_release_date(form_type, normalized_record.get("filing_release_date"))
            )
            normalized_record["next_release_date_source"] = (
                normalized_record.get("next_release_date_source")
                or "inferred_from_filing_release_date"
            )

        refreshed_records.append(normalized_record)

    return refreshed_records


def _normalize_filing_release_metadata(form_type, filing_record):
    normalized_record = dict(filing_record or {})
    filing_release_date = (
        _format_date_like(normalized_record.get("filing_release_date"))
        or _format_date_like(normalized_record.get("release_date"))
        or _format_date_like(normalized_record.get("filing_date"))
    )
    next_release_date = (
        _format_date_like(normalized_record.get("next_release_date"))
        or _format_date_like(normalized_record.get("next_filing_date"))
        or _format_date_like(normalized_record.get("expected_next_release_date"))
    )

    if next_release_date:
        next_release_source = normalized_record.get("next_release_date_source") or "document_provided"
    else:
        next_release_date = _infer_next_release_date(form_type, filing_release_date)
        next_release_source = "inferred_from_filing_release_date" if next_release_date else None

    normalized_record["filing_release_date"] = filing_release_date
    normalized_record["next_release_date"] = next_release_date
    normalized_record["next_release_date_source"] = next_release_source
    return normalized_record


def _load_persisted_filing_payloads(ticker, form_type, filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR):
    form_dir = _filing_form_dir(ticker, form_type, filings_base_dir=filings_base_dir)
    if not os.path.isdir(form_dir):
        return []

    file_records = []
    for existing_name in os.listdir(form_dir):
        if not existing_name.endswith(".json"):
            continue
        file_path = os.path.join(form_dir, existing_name)
        try:
            with open(file_path, "r", encoding="utf-8") as filing_file:
                filing_record = json.load(filing_file)
        except (OSError, json.JSONDecodeError):
            continue
        normalized_record = dict(filing_record)
        normalized_record["_persisted_file_path"] = file_path
        file_records.append(normalized_record)

    finalized_records = _finalize_filing_records(
        ticker,
        form_type,
        _refresh_filing_schedule_metadata(form_type, _dedupe_filing_records(file_records)),
    )
    normalized_records = []
    for finalized_record in finalized_records:
        normalized_record = dict(finalized_record)
        file_path = normalized_record.pop("_persisted_file_path", None)
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as filing_file:
                    json.dump(normalized_record, filing_file, ensure_ascii=False, indent=2)
            except OSError:
                pass
        normalized_records.append(normalized_record)
    return normalized_records


def _filing_form_is_fresh(form_type, filing_records, as_of_date=None):
    if not filing_records:
        return False

    as_of_date = pd.Timestamp(as_of_date or datetime.now().date()).normalize()
    latest_record = _sorted_filing_records(filing_records)[:1]
    if not latest_record:
        return False

    next_release_date = pd.to_datetime(latest_record[0].get("next_release_date"), errors="coerce")
    if pd.isna(next_release_date):
        return False

    next_release_date = next_release_date.normalize()
    return as_of_date < next_release_date


def _documents_from_payloads(document_payloads):
    return [
        Document(text=payload["text"], metadata=payload["metadata"])
        for payload in document_payloads
    ]


VECTOR_INDEX_METADATA_ALLOWED_KEYS = {
    "ticker",
    "type",
    "form_type",
    "section_key",
    "section_title",
    "company_name",
    "sector",
    "industry",
    "filing_date",
    "reporting_date",
    "fiscal_year",
    "fiscal_period",
    "fiscal_quarter",
    "retrieval_tier",
    "filing_rank",
    "frequency",
    "period_end_date",
    "most_recent",
    "group",
    "subgroup",
    "glossary_domain",
}


def _normalize_metadata_for_vector_index(metadata, max_value_chars=180):
    normalized_metadata = {}
    for key, value in (metadata or {}).items():
        if key not in VECTOR_INDEX_METADATA_ALLOWED_KEYS or value is None:
            continue

        if isinstance(value, list):
            if not value:
                continue
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
            if not value:
                continue
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            value = str(value).strip()

        if not value:
            continue
        if len(value) > max_value_chars:
            value = value[: max_value_chars - 3].rstrip() + "..."
        normalized_metadata[key] = value

    return normalized_metadata


def _prepare_documents_for_vector_index(documents):
    prepared_documents = []
    for document in documents:
        prepared_documents.append(
            Document(
                text=document.text,
                metadata=_normalize_metadata_for_vector_index(document.metadata),
            )
        )
    return prepared_documents


def _load_financial_history_frames(conn, ticker):
    ticker = ticker.upper()
    history = {}
    for frequency in ("Annual", "Quarterly"):
        df = pd.read_sql_query(
            """
            SELECT *
            FROM financial_indicators
            WHERE Ticker = ? AND Frequency = ?
            ORDER BY date([Period End Date]) DESC
            """,
            conn,
            params=(ticker, frequency),
        )
        if df.empty:
            history[frequency] = df
            continue

        df["Period End Date"] = pd.to_datetime(df["Period End Date"], errors="coerce")
        df = (
            df.dropna(subset=["Period End Date"])
            .sort_values("Period End Date", ascending=False)
            .reset_index(drop=True)
        )
        history[frequency] = df
    return history


def _match_filing_row(financial_history, filing_record):
    frequency = "Annual" if filing_record["form_type"] == "10-K" else "Quarterly"
    df = financial_history.get(frequency)
    if df is None or df.empty:
        return frequency, None, None

    report_date = pd.to_datetime(filing_record.get("reporting_date"), errors="coerce")
    if pd.isna(report_date):
        report_date = pd.to_datetime(filing_record.get("filing_date"), errors="coerce")

    if pd.notna(report_date):
        exact_match = df[df["Period End Date"] == report_date]
        if not exact_match.empty:
            row_index = int(exact_match.index[0])
        else:
            delta_days = (df["Period End Date"] - report_date).abs().dt.days
            row_index = int(delta_days.idxmin())
    else:
        row_index = 0

    row = df.iloc[row_index]
    previous_row = df.iloc[row_index + 1] if row_index + 1 < len(df) else None
    return frequency, row, previous_row


def _row_value(row, column_name):
    if row is None or column_name not in row.index:
        return None
    value = row.get(column_name)
    if pd.isna(value):
        return None
    return value


def _format_compact_number(value, prefix=""):
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return None
    absolute_value = abs(numeric_value)
    if absolute_value >= 1_000_000_000_000:
        return f"{prefix}{numeric_value / 1_000_000_000_000:.2f}T"
    if absolute_value >= 1_000_000_000:
        return f"{prefix}{numeric_value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"{prefix}{numeric_value / 1_000_000:.2f}M"
    if absolute_value >= 1_000:
        return f"{prefix}{numeric_value / 1_000:.2f}K"
    return f"{prefix}{numeric_value:.2f}"


def _format_indicator_value(indicator_name, value):
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return None

    percent_markers = (
        "margin",
        "growth",
        "yield",
        "payout",
        "return on",
        "cfroi",
    )
    name = indicator_name.lower()
    if any(marker in name for marker in percent_markers):
        return f"{numeric_value:.2f}%"
    if "cycle" in name:
        return f"{numeric_value:.1f} days"
    return f"{numeric_value:.2f}"


def _format_statement_fact_value(column_name, value):
    numeric_value = _coerce_numeric(value)
    if numeric_value is None:
        return None

    if column_name == "Diluted EPS":
        return _format_compact_number(numeric_value, prefix="$")
    if column_name == "Shares Outstanding":
        return _format_compact_number(numeric_value)
    return _format_compact_number(numeric_value, prefix="$")


def _change_fragment(label, current_value, previous_value, style):
    current_numeric = _coerce_numeric(current_value)
    previous_numeric = _coerce_numeric(previous_value)
    if current_numeric is None or previous_numeric is None:
        return None

    if style == "pct":
        if previous_numeric == 0:
            return None
        change_pct = ((current_numeric - previous_numeric) / abs(previous_numeric)) * 100.0
        direction = "up" if change_pct >= 0 else "down"
        return f"{label} {direction} {abs(change_pct):.1f}%"

    if style == "bps":
        change_bps = (current_numeric - previous_numeric) * 100.0
        direction = "up" if change_bps >= 0 else "down"
        return f"{label} {direction} {abs(change_bps):.0f} bps"

    return None


def _filing_change_summary(frequency, current_row, previous_row):
    if current_row is None or previous_row is None:
        return ""

    summary_parts = []
    for column_name, label, style in [
        ("Total Revenue", "revenue", "pct"),
        ("Gross Margin", "gross margin", "bps"),
        ("Operating Margin", "operating margin", "bps"),
        ("Net Profit Margin", "net margin", "bps"),
        ("Diluted EPS", "diluted EPS", "pct"),
        ("Free Cash Flow", "free cash flow", "pct"),
    ]:
        fragment = _change_fragment(
            label,
            _row_value(current_row, column_name),
            _row_value(previous_row, column_name),
            style,
        )
        if fragment:
            summary_parts.append(fragment)

    if not summary_parts:
        return ""

    comparison_label = "prior annual period" if frequency == "Annual" else "prior quarterly period"
    return "; ".join(summary_parts[:4]) + f" versus the {comparison_label}."


def _statement_fact_lines_from_row(row):
    if row is None:
        return []

    fact_lines = []
    for column_name, label in STATEMENT_FACT_COLUMNS:
        formatted_value = _format_statement_fact_value(column_name, _row_value(row, column_name))
        if formatted_value:
            fact_lines.append(f"{label}: {formatted_value}")

    current_assets = _coerce_numeric(_row_value(row, "Current Assets"))
    current_liabilities = _coerce_numeric(_row_value(row, "Current Liabilities"))
    if current_assets is not None and current_liabilities is not None:
        working_capital = current_assets - current_liabilities
        fact_lines.append(f"Working Capital: {_format_compact_number(working_capital, prefix='$')}")

    return fact_lines


def _extract_matching_passages(text, spec_list, max_chars=340):
    if not text:
        return []

    blocks = [
        re.sub(r"\s+", " ", block).strip()
        for block in re.split(r"\n{2,}", _clean_section_text(text))
        if len(block.strip()) >= 80
    ]
    if not blocks:
        blocks = [
            re.sub(r"\s+", " ", block).strip()
            for block in re.split(r"(?<=[.!?])\s+", _clean_section_text(text))
            if len(block.strip()) >= 80
        ]

    matched_lines = []
    for spec in spec_list:
        matches = []
        for block in blocks:
            lower_block = block.lower()
            required_terms = spec.get("all", [])
            optional_terms = spec.get("any", [])
            if required_terms and not all(term in lower_block for term in required_terms):
                continue
            if optional_terms and not any(term in lower_block for term in optional_terms):
                continue
            snippet = _truncate_text(block, max_chars=max_chars)
            if snippet in matches:
                continue
            matches.append(snippet)
            if len(matches) >= spec.get("max_matches", 2):
                break
        if matches:
            matched_lines.append(f"{spec['label']}: {' | '.join(matches)}")

    return matched_lines


def _is_financial_sector(company_profile):
    sector_blob = " ".join(
        [
            str(company_profile.get("sector") or ""),
            str(company_profile.get("industry") or ""),
        ]
    ).lower()
    return any(
        keyword in sector_blob
        for keyword in [
            "bank",
            "insurance",
            "financial",
            "reit",
            "broker",
            "capital markets",
            "asset management",
            "investment",
        ]
    )


def _financial_statement_documents_from_filing(ticker, company_profile, filing_record, financial_history):
    form_type = filing_record["form_type"]
    section_key, section_title = FILING_STATEMENT_SECTIONS[form_type]
    section_text = (filing_record.get("sections") or {}).get(section_key) or ""

    frequency, matched_row, previous_row = _match_filing_row(financial_history, filing_record)
    change_summary = _filing_change_summary(frequency, matched_row, previous_row)
    fact_lines = _statement_fact_lines_from_row(matched_row)
    statement_linked_lines = _extract_matching_passages(section_text, STATEMENT_LINKED_FACT_SPECS)
    financial_sector_lines = (
        _extract_matching_passages(section_text, FINANCIAL_SECTOR_NOTE_SPECS)
        if _is_financial_sector(company_profile)
        else []
    )

    base_metadata = _filing_base_metadata(
        ticker,
        company_profile,
        filing_record,
        doc_type="filing_financial_summary",
        section_key=section_key,
        section_title=section_title,
    )
    documents = []

    if matched_row is not None:
        indicator_lines = []
        for indicator_name in CORE_GLOSSARY_INDICATORS:
            formatted_value = _format_indicator_value(indicator_name, _row_value(matched_row, indicator_name))
            if formatted_value is None:
                continue
            indicator_lines.append(f"{indicator_name}: {formatted_value}")

        if indicator_lines:
            documents.append(
                Document(
                    text="\n".join(
                        [
                            f"**{section_title} Derived Indicators**",
                            "",
                            f"Company: {company_profile['company_name']}",
                            f"Ticker: {ticker}",
                            f"Form Type: {form_type}",
                            f"Filing Date: {filing_record['filing_date']}",
                            f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}",
                            f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}",
                            f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}",
                            f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}",
                            "Document Role: Derived 48-indicator filing snapshot from structured financial data.",
                            "",
                            f"Short change summary: {change_summary or 'No automatic change summary available.'}",
                            "",
                            "Derived indicators:",
                            *indicator_lines,
                        ]
                    ),
                    metadata=base_metadata
                    | {
                        "type": "filing_derived_indicators",
                        "source": f"{section_title} derived indicators",
                        "frequency": frequency,
                        "period_end_date": str(matched_row["Period End Date"].date()),
                    },
                )
            )

        if fact_lines:
            documents.append(
                Document(
                    text="\n".join(
                        [
                            f"**{section_title} Reported Facts**",
                            "",
                            f"Company: {company_profile['company_name']}",
                            f"Ticker: {ticker}",
                            f"Form Type: {form_type}",
                            f"Filing Date: {filing_record['filing_date']}",
                            f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}",
                            f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}",
                            f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}",
                            f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}",
                            "Document Role: Key reported facts tied to the filing-period financial statements.",
                            "",
                            f"Short change summary: {change_summary or 'No automatic change summary available.'}",
                            "",
                            "Reported facts:",
                            *fact_lines,
                        ]
                    ),
                    metadata=base_metadata
                    | {
                        "type": "filing_reported_facts",
                        "source": f"{section_title} reported facts",
                        "frequency": frequency,
                        "period_end_date": str(matched_row["Period End Date"].date()),
                    },
                )
            )

    if statement_linked_lines:
        documents.append(
            Document(
                text="\n".join(
                    [
                        f"**{section_title} Statement-Linked Facts**",
                        "",
                        f"Company: {company_profile['company_name']}",
                        f"Ticker: {ticker}",
                        f"Form Type: {form_type}",
                        f"Filing Date: {filing_record['filing_date']}",
                        f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}",
                        f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}",
                        f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}",
                        f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}",
                        "Document Role: Statement-linked facts extracted from financial statement notes and disclosures.",
                        "",
                        *statement_linked_lines,
                    ]
                ),
                metadata=base_metadata
                | {
                    "type": "statement_linked_facts",
                    "source": f"{section_title} statement-linked facts",
                },
            )
        )

    if financial_sector_lines:
        documents.append(
            Document(
                text="\n".join(
                    [
                        f"**{section_title} Financial-Sector Note Summary**",
                        "",
                        f"Company: {company_profile['company_name']}",
                        f"Ticker: {ticker}",
                        f"Form Type: {form_type}",
                        f"Filing Date: {filing_record['filing_date']}",
                        f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}",
                        f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}",
                        f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}",
                        f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}",
                        "Document Role: Richer note summary for financial-sector disclosures.",
                        "",
                        *financial_sector_lines,
                    ]
                ),
                metadata=base_metadata
                | {
                    "type": "financial_sector_note_summary",
                    "source": f"{section_title} financial-sector note summary",
                },
            )
        )

    summary_fragments = []
    if change_summary:
        summary_fragments.append(change_summary)
    if fact_lines:
        summary_fragments.append("Key facts: " + "; ".join(fact_lines[:4]))
    if statement_linked_lines:
        summary_fragments.append("Linked facts: " + "; ".join(statement_linked_lines[:2]))
    if financial_sector_lines:
        summary_fragments.append("Financial-sector notes: " + "; ".join(financial_sector_lines[:2]))

    if summary_fragments:
        documents.append(
            Document(
                text="\n".join(
                    [
                        f"**{section_title} Summary**",
                        "",
                        f"Company: {company_profile['company_name']}",
                        f"Ticker: {ticker}",
                        f"Form Type: {form_type}",
                        f"Filing Date: {filing_record['filing_date']}",
                        f"Reporting Date: {filing_record.get('reporting_date') or 'Unknown'}",
                        f"Fiscal Year: {filing_record.get('fiscal_year') or 'Unknown'}",
                        f"Fiscal Period: {filing_record.get('fiscal_period') or 'Unknown'}",
                        f"Retrieval Tier: {filing_record.get('retrieval_tier', 'archive')}",
                        "",
                        "Short summary:",
                        _truncate_text(" ".join(summary_fragments), max_chars=900),
                    ]
                ),
                metadata=base_metadata
                | {
                    "type": "filing_financial_summary",
                    "source": f"{section_title} summary",
                    "section_summary": _truncate_text(" ".join(summary_fragments), max_chars=900),
                },
            )
        )

    return documents


def _build_company_profile_from_sec_archive(ticker, archive_payload):
    latest_10k = (archive_payload.get("10k_filings") or [{}])[0]
    latest_10q = (archive_payload.get("10q_filings") or [{}])[0]
    yahoo_fallback = _get_yahoo_taxonomy_fallback(ticker)

    company_name = (
        latest_10k.get("company_name")
        or archive_payload.get("company_name")
        or yahoo_fallback.get("company_name")
        or ticker
    )
    sector = _derive_sector_from_sic(
        latest_10k.get("sic"),
        latest_10k.get("sic_description"),
    ) or yahoo_fallback.get("sector")
    industry = _derive_industry_from_sic(
        latest_10k.get("sic"),
        latest_10k.get("sic_description"),
    ) or yahoo_fallback.get("industry")

    item_1_text = (latest_10k.get("sections") or {}).get("item_1_business") or ""
    description = _truncate_text(item_1_text, max_chars=3200)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "sector_source": "SEC SIC-derived sector" if sector else None,
        "industry_source": "SEC SIC-derived industry" if industry else None,
        "peer_tickers": [],
        "peer_source": None,
        "description": description,
        "source": "SEC EDGAR filings archive" if latest_10k else "Yahoo Finance via yfinance",
        "cik": archive_payload.get("cik"),
        "sic": latest_10k.get("sic") or archive_payload.get("sic"),
        "sic_description": latest_10k.get("sic_description") or archive_payload.get("sic_description"),
        "latest_filing_date": latest_10k.get("filing_date"),
        "latest_filing_url": latest_10k.get("filing_url"),
        "latest_10k_release_date": latest_10k.get("filing_release_date") or latest_10k.get("filing_date"),
        "latest_10k_next_release_date": latest_10k.get("next_release_date"),
        "latest_10k_next_release_source": latest_10k.get("next_release_date_source"),
        "latest_10k_fiscal_year_end": latest_10k.get("fiscal_year_end") or latest_10k.get("reporting_date"),
        "latest_10k_filer_status": latest_10k.get("filer_status"),
        "latest_10k_filer_status_label": _humanize_filer_status(latest_10k.get("filer_status")),
        "latest_10k_filing_deadline_days": latest_10k.get("filing_deadline_days"),
        "latest_10k_filing_lag_days": latest_10k.get("filing_lag_days"),
        "latest_10k_historical_filing_lag_days": latest_10k.get("historical_filing_lag_days"),
        "latest_10q_release_date": latest_10q.get("filing_release_date") or latest_10q.get("filing_date"),
        "latest_10q_next_release_date": latest_10q.get("next_release_date"),
        "latest_10q_next_release_source": latest_10q.get("next_release_date_source"),
        "ten_k_filings": archive_payload.get("10k_filings", []),
        "ten_q_filings": archive_payload.get("10q_filings", []),
    }


def _archive_payload_from_local_files(ticker, filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR):
    ticker = ticker.upper()
    ten_k_filings = _load_persisted_filing_payloads(ticker, "10-K", filings_base_dir=filings_base_dir)
    ten_q_filings = _load_persisted_filing_payloads(ticker, "10-Q", filings_base_dir=filings_base_dir)
    reference_filer_status = next(
        (
            filing_record.get("filer_status")
            for filing_record in ten_k_filings + ten_q_filings
            if filing_record.get("filer_status")
        ),
        None,
    )
    reference_fiscal_year_end = next(
        (
            filing_record.get("reporting_date") or filing_record.get("fiscal_year_end")
            for filing_record in ten_k_filings
            if filing_record.get("reporting_date") or filing_record.get("fiscal_year_end")
        ),
        None,
    )
    ten_k_filings = _finalize_filing_records(
        ticker,
        "10-K",
        _refresh_filing_schedule_metadata("10-K", _roll_filing_window(ten_k_filings, max_records=10)),
    )
    ten_q_filings = _finalize_filing_records(
        ticker,
        "10-Q",
        _refresh_filing_schedule_metadata(
            "10-Q",
            _roll_filing_window(ten_q_filings, max_records=12),
            reference_filer_status=reference_filer_status,
            reference_fiscal_year_end=reference_fiscal_year_end,
        ),
    )
    _persist_filing_payloads(ticker, "10-K", ten_k_filings, filings_base_dir=filings_base_dir, max_records=10)
    _persist_filing_payloads(ticker, "10-Q", ten_q_filings, filings_base_dir=filings_base_dir, max_records=12)
    local_records = _sorted_filing_records(ten_k_filings + ten_q_filings)
    latest_record = local_records[0] if local_records else {}

    return {
        "ticker": ticker,
        "company_name": latest_record.get("company_name") or ticker,
        "cik": latest_record.get("cik"),
        "sic": latest_record.get("sic"),
        "sic_description": latest_record.get("sic_description"),
        "10k_filings": ten_k_filings,
        "10q_filings": ten_q_filings,
    }


def _sync_sec_filing_archive(
    ticker,
    filings_base_dir=DEFAULT_STOCK_FILINGS_BASE_DIR,
    max_10k_filings=10,
    max_10q_filings=12):
    ticker = ticker.upper()
    local_10k_filings = _load_persisted_filing_payloads(ticker, "10-K", filings_base_dir=filings_base_dir)
    local_10q_filings = _load_persisted_filing_payloads(ticker, "10-Q", filings_base_dir=filings_base_dir)

    ten_k_is_fresh = _filing_form_is_fresh("10-K", local_10k_filings)
    ten_q_is_fresh = _filing_form_is_fresh("10-Q", local_10q_filings)

    if ten_k_is_fresh and ten_q_is_fresh:
        return _archive_payload_from_local_files(ticker, filings_base_dir=filings_base_dir)

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
        "industry": _derive_industry_from_sic(sic, sic_description),
    }

    filing_groups = {
        "10-K": _roll_filing_window(local_10k_filings, max_records=max_10k_filings) if ten_k_is_fresh else [],
        "10-Q": _roll_filing_window(local_10q_filings, max_records=max_10q_filings) if ten_q_is_fresh else [],
    }
    max_by_form = {"10-K": max_10k_filings, "10-Q": max_10q_filings}
    cik_no_zero = str(int(cik))

    forms = recent.get("form") or []
    accession_numbers = recent.get("accessionNumber") or []
    filing_dates = recent.get("filingDate") or []
    primary_documents = recent.get("primaryDocument") or []
    report_dates = recent.get("reportDate") or []
    fiscal_years = recent.get("fy") or []
    fiscal_periods = recent.get("fp") or []

    for row_index, form in enumerate(forms):
        accession_number = accession_numbers[row_index] if row_index < len(accession_numbers) else None
        filing_date = filing_dates[row_index] if row_index < len(filing_dates) else None
        primary_document = primary_documents[row_index] if row_index < len(primary_documents) else None
        report_date = report_dates[row_index] if row_index < len(report_dates) else None
        fiscal_year = fiscal_years[row_index] if row_index < len(fiscal_years) else None
        fiscal_period = fiscal_periods[row_index] if row_index < len(fiscal_periods) else None

        if form not in filing_groups or len(filing_groups[form]) >= max_by_form[form]:
            continue
        if form == "10-K" and ten_k_is_fresh:
            continue
        if form == "10-Q" and ten_q_is_fresh:
            continue
        if not accession_number or not filing_date or not primary_document:
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
        filer_status = _extract_filer_status(filing_text)

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
            "reporting_date": report_date,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "filer_status": filer_status,
            "recency_rank": len(filing_groups[form]),
            "retrieval_tier": _filing_retrieval_tier(form, len(filing_groups[form])),
            "sections": sections,
        }
        filing_groups[form] = _roll_filing_window(
            filing_groups[form] + [filing_record],
            max_records=max_by_form[form],
        )

    reference_filer_status = next(
        (
            filing_record.get("filer_status")
            for filing_record in filing_groups["10-K"] + filing_groups["10-Q"]
            if filing_record.get("filer_status")
        ),
        None,
    )
    reference_fiscal_year_end = next(
        (
            filing_record.get("reporting_date") or filing_record.get("fiscal_year_end")
            for filing_record in filing_groups["10-K"]
            if filing_record.get("reporting_date") or filing_record.get("fiscal_year_end")
        ),
        None,
    )
    for form_type in ("10-K", "10-Q"):
        rolled_records = _roll_filing_window(filing_groups[form_type], max_records=max_by_form[form_type])
        refreshed_records = _refresh_filing_schedule_metadata(
            form_type,
            rolled_records,
            reference_filer_status=reference_filer_status if form_type == "10-Q" else None,
            reference_fiscal_year_end=reference_fiscal_year_end if form_type == "10-Q" else None,
        )
        filing_groups[form_type] = _finalize_filing_records(ticker, form_type, refreshed_records)

    _persist_filing_payloads(
        ticker,
        "10-K",
        filing_groups["10-K"],
        filings_base_dir=filings_base_dir,
        max_records=max_10k_filings,
    )
    _persist_filing_payloads(
        ticker,
        "10-Q",
        filing_groups["10-Q"],
        filings_base_dir=filings_base_dir,
        max_records=max_10q_filings,
    )

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
    market_snapshot=None):
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
            f"CREATE TABLE IF NOT EXISTS financial_indicators ({', '.join(column_definitions)})")
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
    ten_k_filings = _sorted_filing_records(company_profile.get("ten_k_filings", []))
    ten_q_filings = _sorted_filing_records(company_profile.get("ten_q_filings", []))
    financial_history = _load_financial_history_frames(conn, ticker)

    profile_lines = [
        f"Company: {company_profile['company_name']}",
        f"Ticker: {ticker}",
        f"Industry: {company_profile['industry'] or 'Unknown'}",
        f"Sector: {company_profile['sector'] or 'Unknown'}",
        f"Industry Source: {company_profile.get('industry_source') or 'Unknown'}",
        f"Sector Source: {company_profile.get('sector_source') or 'Unknown'}",
        f"Source: {company_profile['source']}",
        f"Stored 10-K filings: {len(ten_k_filings)}",
        f"Stored 10-Q filings: {len(ten_q_filings)}",
        f"Active full-text 10-K filings: {min(ACTIVE_FULLTEXT_FILING_LIMITS['10-K'], len(ten_k_filings))}",
        f"Active full-text 10-Q filings: {min(ACTIVE_FULLTEXT_FILING_LIMITS['10-Q'], len(ten_q_filings))}",
    ]
    if company_profile.get("latest_filing_date"):
        profile_lines.append(f"Latest 10-K Filing Date: {company_profile['latest_filing_date']}")
    if company_profile.get("latest_filing_url"):
        profile_lines.append(f"Latest 10-K Filing URL: {company_profile['latest_filing_url']}")
    if company_profile.get("latest_10k_release_date"):
        profile_lines.append(f"Latest 10-K Release Date: {company_profile['latest_10k_release_date']}")
    if company_profile.get("latest_10k_fiscal_year_end"):
        profile_lines.append(f"Latest 10-K Fiscal Year End: {company_profile['latest_10k_fiscal_year_end']}")
    if company_profile.get("latest_10k_filer_status_label"):
        profile_lines.append(f"Latest 10-K Filer Status: {company_profile['latest_10k_filer_status_label']}")
    if company_profile.get("latest_10k_historical_filing_lag_days") is not None:
        profile_lines.append(
            f"Historical 10-K Filing Lag Used: {company_profile['latest_10k_historical_filing_lag_days']} days"
        )
    if company_profile.get("latest_10k_filing_deadline_days") is not None:
        profile_lines.append(
            f"Estimated 10-K SEC Deadline: {company_profile['latest_10k_filing_deadline_days']} days after fiscal year end"
        )
    if company_profile.get("latest_10k_next_release_date"):
        profile_lines.append(
            f"Next 10-K Release Date: {company_profile['latest_10k_next_release_date']} "
            f"({company_profile.get('latest_10k_next_release_source') or 'stored'})"
        )
    if company_profile.get("latest_10q_release_date"):
        profile_lines.append(f"Latest 10-Q Release Date: {company_profile['latest_10q_release_date']}")
    if company_profile.get("latest_10q_next_release_date"):
        profile_lines.append(
            f"Next 10-Q Release Date: {company_profile['latest_10q_next_release_date']} "
            f"({company_profile.get('latest_10q_next_release_source') or 'stored'})"
        )
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
                "sector_source": company_profile.get("sector_source"),
                "industry_source": company_profile.get("industry_source"),
                "peer_tickers": company_profile.get("peer_tickers", []),
                "peer_count": len(company_profile.get("peer_tickers", [])),
                "peer_source": company_profile.get("peer_source"),
                "source": company_profile["source"],
                "filing_date": company_profile.get("latest_filing_date"),
                "filing_url": company_profile.get("latest_filing_url"),
                "latest_10k_release_date": company_profile.get("latest_10k_release_date"),
                "latest_10k_next_release_date": company_profile.get("latest_10k_next_release_date"),
                "latest_10k_next_release_source": company_profile.get("latest_10k_next_release_source"),
                "latest_10k_fiscal_year_end": company_profile.get("latest_10k_fiscal_year_end"),
                "latest_10k_filer_status": company_profile.get("latest_10k_filer_status"),
                "latest_10k_filer_status_label": company_profile.get("latest_10k_filer_status_label"),
                "latest_10k_filing_deadline_days": company_profile.get("latest_10k_filing_deadline_days"),
                "latest_10k_filing_lag_days": company_profile.get("latest_10k_filing_lag_days"),
                "latest_10k_historical_filing_lag_days": company_profile.get("latest_10k_historical_filing_lag_days"),
                "latest_10q_release_date": company_profile.get("latest_10q_release_date"),
                "latest_10q_next_release_date": company_profile.get("latest_10q_next_release_date"),
                "latest_10q_next_release_source": company_profile.get("latest_10q_next_release_source"),
                "ten_k_count": len(ten_k_filings),
                "ten_q_count": len(ten_q_filings),
            }
        )
    )

    for filing_record in ten_k_filings:
        documents.extend(_documents_from_payloads(filing_record.get("documents", [])))
        documents.extend(
            _financial_statement_documents_from_filing(
                ticker,
                company_profile,
                filing_record,
                financial_history,
            )
        )

    for filing_record in ten_q_filings:
        documents.extend(_documents_from_payloads(filing_record.get("documents", [])))
        documents.extend(
            _financial_statement_documents_from_filing(
                ticker,
                company_profile,
                filing_record,
                financial_history,
            )
        )

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
        quick_summary = _filing_change_summary(freq, df.iloc[0], df.iloc[1] if len(df) > 1 else None)

        text = (
            f"**Latest {len(df)} {freq} Financial Indicators - {ticker}**\n\n"
            f"Company: {company_profile['company_name']}\n"
            f"Industry: {company_profile['industry'] or 'Unknown'}\n"
            f"Sector: {company_profile['sector'] or 'Unknown'}\n"
            f"Latest 10-K Next Release Date: {company_profile.get('latest_10k_next_release_date') or 'Unavailable'}\n"
            f"Latest 10-Q Next Release Date: {company_profile.get('latest_10q_next_release_date') or 'Unavailable'}\n\n"
            f"Current market price used for market-based ratios: {df['Current Market Price'].iloc[0] if 'Current Market Price' in df.columns else 'Unavailable'}\n"
            f"Most recent: {df['Period End Date'].iloc[0]}\n"
            f"Oldest shown: {df['Period End Date'].iloc[-1]}\n\n"
            f"Quick change summary: {quick_summary or 'No automatic change summary available.'}\n\n"
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
                "latest_10k_next_release_date": company_profile.get("latest_10k_next_release_date"),
                "latest_10k_next_release_source": company_profile.get("latest_10k_next_release_source"),
                "latest_10q_next_release_date": company_profile.get("latest_10q_next_release_date"),
                "latest_10q_next_release_source": company_profile.get("latest_10q_next_release_source"),
                "frequency": freq,
                "periods": len(df),
                "most_recent": df['Period End Date'].iloc[0],
                "source": "SEC Company Facts + current market price + SQLite",
                "table_rows": len(df),
                "ten_k_count": len(ten_k_filings),
                "ten_q_count": len(ten_q_filings),
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
      storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
      refresh_graph=True):
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
    index_docs = _prepare_documents_for_vector_index(docs)
    node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=70)
    nodes = node_parser.get_nodes_from_documents(index_docs)
    _reset_persist_dir(persist_dir)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)

    import ingest_graph
    import ingest_macro

    if refresh_graph:
        try:
            ingest_graph.refresh_property_graph_for_ticker(
                ticker,
                stock_docs=docs,
                stock_db_path=db_path,
                macro_db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
                filings_base_dir=filings_base_dir,
            )
        except Exception as exc:
            print(f"Graph layer refresh skipped for {ticker}: {exc}")

    try:
        import analysis

        analysis_result = analysis.get_or_create_daily_benchmark_analysis(
            ticker,
            generate_plots=False,
            persist_to_graph=True,
        )
        print(f"Benchmark analysis summary stored for {ticker}: {analysis_result['conclusion']}")
    except Exception as exc:
        print(f"Benchmark analysis skipped for {ticker}: {exc}")

    print(f"Index successfully refreshed for {ticker}")


#refresh_ticker_data_and_index('aapl')
