import os
import re
import stat
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from html import unescape
from io import StringIO

import ingest_stock
import pandas as pd
import requests
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

load_dotenv("config.env")
MACRO_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS macro_indicators (
        indicator_key TEXT,
        indicator_name TEXT,
        category TEXT,
        release_name TEXT,
        source TEXT,
        series_id TEXT,
        frequency TEXT,
        observation_date TEXT,
        value FLOAT,
        units TEXT,
        source_url TEXT,
        notes TEXT,
        retrieved_at TEXT)
    """


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_TABLE_DATA_URL_TEMPLATE = "https://fred.stlouisfed.org/data/{series_id}"
DEFAULT_MACRO_DB_PATH = os.getenv("MACRO_SQL_DB_PATH", "macro_data.db")
DEFAULT_MACRO_STORAGE_DIR = os.getenv("MACRO_STORAGE_DIR", "./storage/macro")
DEFAULT_GLOSSARY_BASE_DIR = os.getenv("GLOSSARY_BASE_DIR", "./data_store/glossary")
DEFAULT_ECO_GLOSSARY_DIR = os.path.join(DEFAULT_GLOSSARY_BASE_DIR, "eco", "raw")

FRED_SERIES_CATALOG = [
        {
            "indicator_key": "fed_funds_rate",
            "indicator_name": "Federal Funds Rate",
            "series_id": "FEDFUNDS",
            "category": "monetary_policy",
            "release_name": "Federal Funds Effective Rate",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Percent",
            "source_url": "https://fred.stlouisfed.org/series/FEDFUNDS",
            "notes": "Effective federal funds rate.",
            "history_limit": 36,
        },
        {
            "indicator_key": "real_gdp",
            "indicator_name": "Real GDP",
            "series_id": "GDPC1",
            "category": "growth",
            "release_name": "Gross Domestic Product",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Quarterly",
            "units": "Billions of Chained 2017 Dollars",
            "source_url": "https://fred.stlouisfed.org/series/GDPC1",
            "notes": "Real gross domestic product.",
            "history_limit": 20,
            "derivations": [
                {
                    "indicator_key": "real_gdp_qoq_pct",
                    "indicator_name": "Real GDP QoQ Change",
                    "method": "pct_change",
                    "periods": 1,
                    "multiplier": 100.0,
                    "units": "Percent",
                    "notes": "Quarter-over-quarter percent change computed from the Real GDP series.",
                    "history_limit": 12,
                }
            ],
        },
        {
            "indicator_key": "cpi_all_items",
            "indicator_name": "Consumer Price Index",
            "series_id": "CPIAUCSL",
            "category": "inflation",
            "release_name": "Consumer Price Index",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Index 1982-1984=100",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "notes": "Headline CPI for all urban consumers.",
            "history_limit": 36,
            "derivations": [
                {
                    "indicator_key": "cpi_inflation_yoy",
                    "indicator_name": "CPI Inflation YoY",
                    "method": "pct_change",
                    "periods": 12,
                    "multiplier": 100.0,
                    "units": "Percent",
                    "notes": "Year-over-year inflation rate computed from headline CPI.",
                    "history_limit": 24,
                }
            ],
        },
        {
            "indicator_key": "unemployment_rate",
            "indicator_name": "Unemployment Rate",
            "series_id": "UNRATE",
            "category": "labor",
            "release_name": "Employment Situation",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Percent",
            "source_url": "https://fred.stlouisfed.org/series/UNRATE",
            "notes": "Civilian unemployment rate.",
            "history_limit": 36,
        },
        {
            "indicator_key": "adp_private_payrolls",
            "indicator_name": "ADP Private Nonfarm Employment",
            "series_id": "ADPMNUSNERSA",
            "category": "labor",
            "release_name": "ADP National Employment Report",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Persons",
            "source_url": "https://fred.stlouisfed.org/series/ADPMNUSNERSA",
            "notes": "Total nonfarm private payroll employment from the ADP report.",
            "history_limit": 36,
            "derivations": [
                {
                    "indicator_key": "adp_private_payrolls_mom_change",
                    "indicator_name": "ADP Private Payrolls MoM Change",
                    "method": "difference",
                    "periods": 1,
                    "multiplier": 1.0,
                    "units": "Persons",
                    "notes": "Month-over-month change in ADP private payroll employment.",
                    "history_limit": 24,
                }
            ],
        },
        {
            "indicator_key": "nonfarm_payrolls",
            "indicator_name": "Total Nonfarm Payrolls",
            "series_id": "PAYEMS",
            "category": "labor",
            "release_name": "Employment Situation",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Thousands of Persons",
            "source_url": "https://fred.stlouisfed.org/series/PAYEMS",
            "notes": "Total nonfarm payroll employment from the BLS Employment Situation release.",
            "history_limit": 36,
            "derivations": [
                {
                    "indicator_key": "nonfarm_payrolls_mom_change",
                    "indicator_name": "Nonfarm Payrolls MoM Change",
                    "method": "difference",
                    "periods": 1,
                    "multiplier": 1.0,
                    "units": "Thousands of Persons",
                    "notes": "Month-over-month change in total nonfarm payrolls.",
                    "history_limit": 24,
                }
            ],
        },
        {
            "indicator_key": "average_hourly_earnings",
            "indicator_name": "Average Hourly Earnings",
            "series_id": "CES0500000003",
            "category": "labor",
            "release_name": "Employment Situation",
            "source": "Federal Reserve Bank of St. Louis (FRED)",
            "frequency": "Monthly",
            "units": "Dollars per Hour",
            "source_url": "https://fred.stlouisfed.org/series/CES0500000003",
            "notes": "Average hourly earnings of all employees, total private.",
            "history_limit": 36,
            "derivations": [
                {
                    "indicator_key": "average_hourly_earnings_yoy",
                    "indicator_name": "Average Hourly Earnings YoY",
                    "method": "pct_change",
                    "periods": 12,
                    "multiplier": 100.0,
                    "units": "Percent",
                    "notes": "Year-over-year wage growth computed from average hourly earnings.",
                    "history_limit": 24,
                }
            ],
        },
    ]


ISM_SERIES_CATALOG = [
        {
            "indicator_key": "pmi",
            "indicator_name": "PMI",
            "category": "business_activity",
            "release_name": "ISM PMI Report",
            "source": "Institute for Supply Management",
            "frequency": "Monthly",
            "units": "Percent",
            "notes": "PMI from the official ISM monthly report page.",
            "sector": "manufacturing",
            "history_limit": 12,
            "ycharts_url": "https://ycharts.com/indicators/us_pmi",
        },
    ]


MACRO_GLOSSARY_FILE_TO_KEYS = {
    "fed interest rate": {"fed_funds_rate"},
    "gdp": {"real_gdp"},
    "cpi": {"cpi_all_items"},
    "inflation rate": {"cpi_inflation_yoy"},
    "unemployment rate": {"unemployment_rate"},
    "adp": {"adp_private_payrolls"},
    "bls": {"nonfarm_payrolls"},
    "pmi": {"pmi"},
}


def _normalize_glossary_indicator_name(name):
    return re.sub(r"\s+", " ", name.strip().lower())


def _available_macro_glossary_names(glossary_dir=DEFAULT_ECO_GLOSSARY_DIR):
    glossary_names = set()
    candidate_dirs = [
        glossary_dir,
        os.path.join(DEFAULT_GLOSSARY_BASE_DIR, "eco"),
    ]

    for candidate_dir in candidate_dirs:
        if not os.path.isdir(candidate_dir):
            continue

        for file_name in os.listdir(candidate_dir):
            file_path = os.path.join(candidate_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            stem, _ = os.path.splitext(file_name)
            glossary_names.add(_normalize_glossary_indicator_name(stem))

    return glossary_names


def _selected_macro_catalogs(glossary_dir=DEFAULT_ECO_GLOSSARY_DIR):
    glossary_names = _available_macro_glossary_names(glossary_dir=glossary_dir)
    allowed_keys = set()

    for glossary_name in glossary_names:
        allowed_keys.update(MACRO_GLOSSARY_FILE_TO_KEYS.get(glossary_name, set()))

    selected_fred_catalog = []
    for series_def in FRED_SERIES_CATALOG:
        include_base = series_def["indicator_key"] in allowed_keys
        selected_derivations = [
            derivation
            for derivation in series_def.get("derivations", [])
            if derivation["indicator_key"] in allowed_keys
        ]
        if not include_base and not selected_derivations:
            continue

        selected_series_def = dict(series_def)
        selected_series_def["_include_base"] = include_base
        selected_series_def["derivations"] = selected_derivations
        selected_fred_catalog.append(selected_series_def)

    selected_ism_catalog = [
        dict(series_def)
        for series_def in ISM_SERIES_CATALOG
        if series_def["indicator_key"] in allowed_keys
    ]

    return selected_fred_catalog, selected_ism_catalog, allowed_keys


def _ensure_tables(conn):
    conn.execute(MACRO_TABLE_SQL)


def ensure_sql_tables(db_path=DEFAULT_MACRO_DB_PATH):
    conn = sqlite3.connect(db_path)
    _ensure_tables(conn)
    conn.commit()
    conn.close()


def _get_fred_api_key():
    load_dotenv("config.env")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing FRED_API_KEY. Add FRED_API_KEY to config.env or your environment "
            "before calling FRED-backed macro updates."
        )
    return api_key


def _request_with_retry(url, *, params=None, timeout=20, attempts=3):
    last_exc = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=timeout,
            )
            if response.status_code >= 500 and attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            if status_code and status_code < 500:
                raise
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise

    raise last_exc


def _observations_to_dataframe(observations, history_limit=None):
    df = pd.DataFrame(observations)
    if df.empty:
        return pd.DataFrame(columns=["observation_date", "value"])

    df = df.rename(columns={"date": "observation_date"})
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = (
        df["value"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace(".", pd.NA)
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["observation_date", "value"])
    df = df.sort_values("observation_date", ascending=False).reset_index(drop=True)

    if history_limit is not None:
        df = df.head(history_limit)
    return df


def _fetch_fred_series_history_from_table_data(series_id, history_limit=None, timeout=20):
    table_data_url = FRED_TABLE_DATA_URL_TEMPLATE.format(series_id=series_id.lower())
    response = _request_with_retry(table_data_url, timeout=timeout)
    text = unescape(response.text)
    matches = re.findall(
        r"(?m)^\s*(\d{4}-\d{2}-\d{2})\s+(-?\d[\d,]*(?:\.\d+)?)\s*$",
        text,
    )

    if not matches:
        raise ValueError(f"Could not parse FRED table data for series {series_id}")

    observations = [{"date": date_text, "value": value_text} for date_text, value_text in matches]
    return _observations_to_dataframe(observations, history_limit=history_limit)


def _fetch_fred_series_history(series_id, history_limit=None, timeout=20):
    api_error = None

    try:
        params = {
            "series_id": series_id,
            "api_key": _get_fred_api_key(),
            "file_type": "json",
            "sort_order": "desc",
        }
        if history_limit is not None:
            params["limit"] = history_limit

        response = _request_with_retry(
            FRED_OBSERVATIONS_URL,
            params=params,
            timeout=timeout,
        )
        payload = response.json()
        observations = payload.get("observations")
        if observations is None:
            raise ValueError(f"Unexpected response while fetching series {series_id}")
        return _observations_to_dataframe(observations, history_limit=history_limit)
    except Exception as exc:
        api_error = exc

    try:
        print(
            f"FRED API request failed for {series_id}; "
            "falling back to FRED Table Data."
        )
        return _fetch_fred_series_history_from_table_data(
            series_id,
            history_limit=history_limit,
            timeout=timeout,
        )
    except Exception as fallback_exc:
        raise RuntimeError(
            f"FRED API fetch failed for {series_id} ({api_error}); "
            f"table data fallback failed ({fallback_exc})"
        ) from fallback_exc


def _derive_series(base_df, derivation):
    if base_df.empty:
        return pd.DataFrame()

    working_df = base_df.sort_values("observation_date", ascending=True).reset_index(drop=True)
    periods = derivation.get("periods", 1)
    multiplier = derivation.get("multiplier", 1.0)

    if derivation.get("method") == "pct_change":
        derived_values = working_df["value"].pct_change(periods=periods) * multiplier
    elif derivation.get("method") == "difference":
        derived_values = working_df["value"].diff(periods=periods) * multiplier
    else:
        raise ValueError(f"Unsupported derivation method: {derivation.get('method')}")

    derived_df = working_df[["observation_date"]].copy()
    derived_df["value"] = derived_values
    derived_df = derived_df.dropna(subset=["value"])
    derived_df = derived_df.sort_values("observation_date", ascending=False).reset_index(drop=True)

    history_limit = derivation.get("history_limit")
    if history_limit is not None:
        derived_df = derived_df.head(history_limit)
    return derived_df


def _flatten_columns(columns):
    flattened = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and str(part).lower() != "nan"]
            flattened.append(" ".join(parts))
        else:
            flattened.append(str(column).strip())
    return flattened


def _expected_indicator_keys():
    _, _, keys = _selected_macro_catalogs()
    return keys


def _candidate_report_months(now=None):
    now = now or datetime.now()
    current_month = now.replace(day=1)
    previous_month = (current_month - timedelta(days=1)).replace(day=1)
    return [current_month, previous_month]


def _ism_report_urls(sector, now=None):
    report_path = "pmi" if sector == "manufacturing" else "services"
    urls = []
    for month_start in _candidate_report_months(now=now):
        month_slug = month_start.strftime("%B").lower()
        urls.append(
            f"https://www.ismworld.org/supply-management-news-and-reports/"
            f"reports/ism-pmi-reports/{report_path}/{month_slug}/"
        )
    return urls


def _html_to_text(html):
    text = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text


def _extract_ism_history_from_text(html, sector, history_limit):
    text = _html_to_text(html)

    sector_phrase = "Manufacturing PMI" if sector == "manufacturing" else "Services PMI"
    marker_patterns = [
        rf"{sector_phrase}.*?Month.*?(?:Average for 12 months|High - Low|Purchasing Managers'? Index)",
        r"THE LAST 12 MONTHS.*?(?:Average for 12 months|High - Low)",
        rf"{sector_phrase} HISTORY.*?(?:Average for 12 months|High - Low)",
    ]

    section = None
    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            section = match.group(0)
            break

    if section is None:
        section = text

    month_pattern = (
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?"
        r")\s+(\d{4})\b"
    )
    month_matches = list(re.finditer(month_pattern, section, flags=re.IGNORECASE))
    rows = []

    for idx, match in enumerate(month_matches):
        start = match.start()
        end = month_matches[idx + 1].start() if idx + 1 < len(month_matches) else len(section)
        chunk = section[start:end]
        month_text = f"{match.group(1)} {match.group(2)}"

        number_match = re.search(r"\b(\d{2}\.\d)\b", chunk)
        if not number_match:
            continue

        rows.append(
            {
                "observation_date": pd.to_datetime(month_text, errors="coerce"),
                "value": float(number_match.group(1)),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).dropna(subset=["observation_date", "value"])
    df = df.drop_duplicates(subset=["observation_date"], keep="first")
    df = df.sort_values("observation_date", ascending=False).reset_index(drop=True)
    if history_limit is not None:
        df = df.head(history_limit)
    return df


def _extract_ism_pmi_history_table_legacy(html, sector, history_limit):
    text = _html_to_text(html)

    if sector == "manufacturing":
        heading_patterns = [
            r"THE LAST 12 MONTHS",
            r"MANUFACTURING PMI HISTORY",
        ]
    else:
        heading_patterns = [
            r"SERVICES PMI HISTORY",
            r"THE LAST 12 MONTHS",
        ]

    history_section = None
    for heading_pattern in heading_patterns:
        match = re.search(
            rf"{heading_pattern}(.*?)(?:Average for 12 months|High\s*[—-]|Low\s*[—-]|About This Report|Business Activity|Manufacturing PMI|Services PMI)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            history_section = match.group(1)
            break

    if history_section is None:
        return pd.DataFrame()

    row_matches = re.findall(
        r"\b("
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
        r")\s+(\d{4})\s*\|\s*(\d{2}\.\d)\b",
        history_section,
        flags=re.IGNORECASE,
    )

    if not row_matches:
        return pd.DataFrame()

    rows = [
        {
            "observation_date": pd.to_datetime(f"{month_text} {year_text}", errors="coerce"),
            "value": float(value_text),
        }
        for month_text, year_text, value_text in row_matches
    ]

    df = pd.DataFrame(rows).dropna(subset=["observation_date", "value"])
    df = df.drop_duplicates(subset=["observation_date"], keep="first")
    df = df.sort_values("observation_date", ascending=False).reset_index(drop=True)
    if history_limit is not None:
        df = df.head(history_limit)
    return df


def _extract_ism_latest_reading_from_text(html, sector):
    text = _html_to_text(html)

    month_match = re.search(
        r"\b("
        r"January|February|March|April|May|June|July|August|September|October|November|December"
        r")\s+(\d{4})\s+ISM",
        text,
        flags=re.IGNORECASE,
    )
    if not month_match:
        return pd.DataFrame()

    if sector == "manufacturing":
        patterns = [
            r"Manufacturing PMI(?:®)?(?:\s+registered|\s+at)?\s+(\d{2}\.\d)",
            r"Purchasing Managers'? Index\s*\(PMI\)\s*(?:registered|at)?\s*(\d{2}\.\d)",
        ]
    else:
        patterns = [
            r"Services PMI(?:®)?(?:\s+registered|\s+at)?\s+(\d{2}\.\d)",
            r"Services PMI(?:®)?\s+at\s+(\d{2}\.\d)",
        ]

    value = None
    for pattern in patterns:
        value_match = re.search(pattern, text, flags=re.IGNORECASE)
        if value_match:
            value = float(value_match.group(1))
            break

    if value is None:
        return pd.DataFrame()

    observation_date = pd.to_datetime(
        f"{month_match.group(1)} {month_match.group(2)}",
        errors="coerce",
    )
    if pd.isna(observation_date):
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "observation_date": observation_date,
                "value": value,
            }
        ]
    )


def _extract_ism_history_from_html(html, history_limit):
    best_df = pd.DataFrame()

    for table in pd.read_html(StringIO(html)):
        if table.empty or len(table.columns) < 2:
            continue

        table = table.copy()
        table.columns = _flatten_columns(table.columns)

        month_col = next(
            (
                column
                for column in table.columns
                if "month" in column.lower()
                or pd.to_datetime(table[column], errors="coerce").notna().sum() >= 6
            ),
            None,
        )
        value_col = next(
            (column for column in table.columns if "pmi" in column.lower()),
            None,
        )

        if not month_col or not value_col:
            continue

        candidate_df = table[[month_col, value_col]].copy()
        candidate_df.columns = ["observation_date", "value"]
        candidate_df["observation_date"] = pd.to_datetime(
            candidate_df["observation_date"],
            errors="coerce",
        )
        candidate_df["value"] = pd.to_numeric(candidate_df["value"], errors="coerce")
        candidate_df = candidate_df.dropna(subset=["observation_date", "value"])

        if candidate_df.empty:
            continue

        candidate_df = candidate_df.sort_values("observation_date", ascending=False).reset_index(drop=True)
        if history_limit is not None:
            candidate_df = candidate_df.head(history_limit)

        if best_df.empty or candidate_df["observation_date"].max() > best_df["observation_date"].max():
            best_df = candidate_df

    return best_df


def _extract_ycharts_history_table(html, history_limit):
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame()

    for table in tables:
        if table.empty:
            continue

        column_map = {str(column).strip().lower(): column for column in table.columns}
        date_column = column_map.get("date")
        value_column = column_map.get("value")

        if not date_column or not value_column:
            continue

        history_df = table[[date_column, value_column]].copy()
        history_df.columns = ["observation_date", "value"]
        history_df["observation_date"] = pd.to_datetime(
            history_df["observation_date"],
            errors="coerce",
        )
        history_df["value"] = pd.to_numeric(history_df["value"], errors="coerce")
        history_df = history_df.dropna(subset=["observation_date", "value"])

        if history_df.empty:
            continue

        history_df = history_df.drop_duplicates(subset=["observation_date"], keep="first")
        history_df = history_df.sort_values("observation_date", ascending=False).reset_index(drop=True)
        if history_limit is not None:
            history_df = history_df.head(history_limit)
        return history_df

    return pd.DataFrame()


def _fetch_ycharts_pmi_history(series_def, timeout=20):
    ycharts_url = series_def.get("ycharts_url")
    if not ycharts_url:
        raise ValueError("No YCharts URL configured.")

    response = requests.get(ycharts_url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    history_df = _extract_ycharts_history_table(response.text,history_limit=series_def.get("history_limit"))

    if history_df.empty:
        raise ValueError("Could not parse YCharts PMI history table.")
    return history_df, ycharts_url


def _extract_ism_pmi_history_table(html, sector, history_limit):
    text = _html_to_text(html)

    if sector == "manufacturing":
        heading_patterns = [r"THE LAST 12 MONTHS",
            r"MANUFACTURING\s+PMI.*?HISTORY"]
    else:
        heading_patterns = [r"SERVICES\s+PMI.*?HISTORY", r"THE LAST 12 MONTHS"]

    history_section = None
    for heading_pattern in heading_patterns:
        match = re.search(
            heading_pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            section_start = match.end()
            trailing_text = text[section_start:]
            average_match = re.search(
                r"Average for 12 months",
                trailing_text,
                flags=re.IGNORECASE,
            )
            if average_match:
                history_section = trailing_text[:average_match.start()]
            else:
                history_section = trailing_text[:1500]
            break

    if history_section is None:
        return pd.DataFrame()

    row_matches = re.findall(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?"
        r")\s+(\d{4})\s+(\d{2}\.\d)\b",
        history_section,
        flags=re.IGNORECASE,
    )

    if not row_matches:
        return pd.DataFrame()

    rows = [
        {
            "observation_date": pd.to_datetime(f"{month_text} {year_text}", errors="coerce"),
            "value": float(value_text),
        }
        for month_text, year_text, value_text in row_matches
    ]

    df = pd.DataFrame(rows).dropna(subset=["observation_date", "value"])
    df = df.drop_duplicates(subset=["observation_date"], keep="first")
    df = df.sort_values("observation_date", ascending=False).reset_index(drop=True)
    if history_limit is not None:
        df = df.head(history_limit)
    return df


def _fetch_ism_pmi_history(series_def, now=None, timeout=20):
    try:
        return _fetch_ycharts_pmi_history(series_def, timeout=timeout)
    except Exception:
        pass

    best_df = pd.DataFrame()
    best_url = None

    for url in _ism_report_urls(series_def["sector"], now=now):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
        except Exception:
            continue

        candidate_df = _extract_ism_pmi_history_table(
            response.text,
            sector=series_def["sector"],
            history_limit=series_def.get("history_limit"),
        )
        if candidate_df.empty:
            try:
                candidate_df = _extract_ism_history_from_html(
                    response.text,
                    history_limit=series_def.get("history_limit"),
                )
            except Exception:
                candidate_df = pd.DataFrame()
        if candidate_df.empty:
            candidate_df = _extract_ism_history_from_text(
                response.text,
                sector=series_def["sector"],
                history_limit=series_def.get("history_limit"),
            )
        if candidate_df.empty:
            candidate_df = _extract_ism_latest_reading_from_text(
                response.text,
                sector=series_def["sector"],
            )
        if candidate_df.empty:
            continue

        if best_df.empty or candidate_df["observation_date"].max() > best_df["observation_date"].max():
            best_df = candidate_df
            best_url = url

    if best_df.empty:
        raise ValueError(f"Could not parse ISM history for {series_def['indicator_name']}")

    return best_df, best_url


def _rows_from_dataframe(df, series_def, retrieved_at):
    rows = df.copy()
    rows["observation_date"] = pd.to_datetime(rows["observation_date"]).dt.strftime("%Y-%m-%d")
    rows["indicator_key"] = series_def["indicator_key"]
    rows["indicator_name"] = series_def["indicator_name"]
    rows["category"] = series_def["category"]
    rows["release_name"] = series_def["release_name"]
    rows["source"] = series_def["source"]
    rows["series_id"] = series_def.get("series_id", series_def["indicator_key"])
    rows["frequency"] = series_def["frequency"]
    rows["units"] = series_def["units"]
    rows["source_url"] = series_def["source_url"]
    rows["notes"] = series_def.get("notes", "")
    rows["retrieved_at"] = retrieved_at

    return rows[
        [
            "indicator_key",
            "indicator_name",
            "category",
            "release_name",
            "source",
            "series_id",
            "frequency",
            "observation_date",
            "value",
            "units",
            "source_url",
            "notes",
            "retrieved_at",
        ]
    ]


def _prune_macro_indicator_rows(conn, allowed_keys):
    if not allowed_keys:
        conn.execute("DELETE FROM macro_indicators")
        return

    placeholders = ",".join("?" for _ in allowed_keys)
    conn.execute(
        f"DELETE FROM macro_indicators WHERE indicator_key NOT IN ({placeholders})",
        tuple(sorted(allowed_keys)),
    )


def market_environment_needs_refresh(db_path=DEFAULT_MACRO_DB_PATH, stale_after_hours=24):
    ensure_sql_tables(db_path)
    expected_indicator_keys = _expected_indicator_keys()
    if not expected_indicator_keys:
        return False

    conn = sqlite3.connect(db_path)
    latest_df = pd.read_sql_query(
        f"""
        SELECT
            MAX(retrieved_at) AS latest_retrieved_at,
            COUNT(DISTINCT indicator_key) AS indicator_count
        FROM macro_indicators
        WHERE indicator_key IN ({",".join("?" for _ in expected_indicator_keys)})
        """,
        conn,
        params=tuple(sorted(expected_indicator_keys)),
    )
    conn.close()

    latest_value = latest_df["latest_retrieved_at"].iloc[0]
    indicator_count = int(latest_df["indicator_count"].iloc[0] or 0)
    if pd.isna(latest_value) or not latest_value:
        return True
    if indicator_count < len(expected_indicator_keys):
        return True

    latest_dt = pd.to_datetime(latest_value, errors="coerce")
    if pd.isna(latest_dt):
        return True

    return datetime.now() - latest_dt.to_pydatetime() >= timedelta(hours=stale_after_hours)


def update_market_environment_records(db_path=DEFAULT_MACRO_DB_PATH):
    """
    Fetches the latest macro-market series, stores them in SQLite, and keeps
    each indicator series fresh for downstream document generation.
    """
    ensure_sql_tables(db_path)
    conn = sqlite3.connect(db_path)
    total_rows_written = 0
    fred_catalog, ism_catalog, allowed_keys = _selected_macro_catalogs()

    _prune_macro_indicator_rows(conn, allowed_keys)

    for series_def in fred_catalog:
        try:
            base_df = _fetch_fred_series_history(
                series_def["series_id"],
                history_limit=series_def.get("history_limit"),
            )
        except Exception as exc:
            print(
                f"Skipping {series_def['indicator_name']} ({series_def['series_id']}): {exc}"
            )
            continue

        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_def = dict(series_def)
        indicator_frames = []
        indicator_keys_to_replace = []

        if series_def.get("_include_base", True):
            base_rows = _rows_from_dataframe(base_df, base_def, retrieved_at)
            indicator_frames.append(base_rows)
            indicator_keys_to_replace.append(base_def["indicator_key"])

        for derivation in series_def.get("derivations", []):
            derived_df = _derive_series(base_df, derivation)
            if derived_df.empty:
                continue

            derived_def = {
                "indicator_key": derivation["indicator_key"],
                "indicator_name": derivation["indicator_name"],
                "category": series_def["category"],
                "release_name": series_def["release_name"],
                "source": series_def["source"],
                "series_id": series_def["series_id"],
                "frequency": series_def["frequency"],
                "units": derivation["units"],
                "source_url": series_def["source_url"],
                "notes": derivation.get("notes", series_def.get("notes", "")),
            }
            indicator_frames.append(_rows_from_dataframe(derived_df, derived_def, retrieved_at))
            indicator_keys_to_replace.append(derivation["indicator_key"])

        if not indicator_frames:
            continue

        conn.executemany(
            "DELETE FROM macro_indicators WHERE indicator_key = ?",
            [(indicator_key,) for indicator_key in indicator_keys_to_replace],
        )

        for frame in indicator_frames:
            frame.to_sql("macro_indicators", conn, if_exists="append", index=False)
            total_rows_written += len(frame)

    for series_def in ism_catalog:
        try:
            history_df, source_url = _fetch_ism_pmi_history(series_def)
        except Exception as exc:
            print(f"Skipping {series_def['indicator_name']}: {exc}")
            continue

        retrieved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pmi_def = dict(series_def)
        pmi_def["source_url"] = source_url or "https://www.ismworld.org/"
        pmi_def["series_id"] = series_def["indicator_key"]
        pmi_rows = _rows_from_dataframe(history_df, pmi_def, retrieved_at)

        conn.execute(
            "DELETE FROM macro_indicators WHERE indicator_key = ?",
            (series_def["indicator_key"],),
        )
        pmi_rows.to_sql("macro_indicators", conn, if_exists="append", index=False)
        total_rows_written += len(pmi_rows)

    conn.commit()
    conn.close()
    return total_rows_written


def refresh_market_environment_if_stale(db_path=DEFAULT_MACRO_DB_PATH, stale_after_hours=24):
    ensure_sql_tables(db_path)
    conn = sqlite3.connect(db_path)
    _prune_macro_indicator_rows(conn, _expected_indicator_keys())
    conn.commit()
    conn.close()

    if market_environment_needs_refresh(
        db_path=db_path,
        stale_after_hours=stale_after_hours,
    ):
        return update_market_environment_records(db_path=db_path)

    print(f"Market environment data is fresh enough; skipping update (< {stale_after_hours}h old).")
    return 0


def _format_macro_value(value, units):
    if pd.isna(value):
        return "NA"

    number = float(value)
    if units == "Percent":
        return f"{number:,.2f}%"
    if units == "Dollars per Hour":
        return f"${number:,.2f}"
    if abs(number) >= 1000:
        return f"{number:,.0f} {units}"
    return f"{number:,.2f} {units}"


def _build_macro_snapshot_df(df):
    snapshot_rows = []

    for _, indicator_df in df.groupby("indicator_key"):
        ordered_df = indicator_df.sort_values("observation_date", ascending=False).reset_index(drop=True)
        latest_row = ordered_df.iloc[0]
        previous_value = None
        change_text = "NA"

        if len(ordered_df) > 1:
            previous_value = ordered_df.iloc[1]["value"]
            change_text = _format_macro_value(
                latest_row["value"] - previous_value,
                latest_row["units"],
            )

        snapshot_rows.append(
            {
                "Indicator": latest_row["indicator_name"],
                "Category": latest_row["category"].replace("_", " ").title(),
                "Release": latest_row["release_name"],
                "Observation Date": latest_row["observation_date"].strftime("%Y-%m-%d"),
                "Latest Value": _format_macro_value(latest_row["value"], latest_row["units"]),
                "Previous Value": (
                    _format_macro_value(previous_value, latest_row["units"])
                    if previous_value is not None
                    else "NA"
                ),
                "Change": change_text,
            }
        )

    return pd.DataFrame(snapshot_rows).sort_values(["Category", "Indicator"]).reset_index(drop=True)


def build_market_environment_docs(db_path=DEFAULT_MACRO_DB_PATH, max_history_per_indicator=12):
    ensure_sql_tables(db_path)
    expected_indicator_keys = _expected_indicator_keys()
    if not expected_indicator_keys:
        return []

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"""
        SELECT *
        FROM macro_indicators
        WHERE indicator_key IN ({",".join("?" for _ in expected_indicator_keys)})
        ORDER BY category, indicator_name, observation_date DESC
        """,
        conn,
        params=tuple(sorted(expected_indicator_keys)),
    )
    conn.close()

    if df.empty:
        return []

    df["observation_date"] = pd.to_datetime(df["observation_date"])
    latest_refresh = pd.to_datetime(df["retrieved_at"], errors="coerce").max()
    latest_refresh_text = latest_refresh.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(latest_refresh) else "Unknown"

    documents = [
        Document(
            text=(
                "**Market Environment Snapshot**\n\n"
                f"Latest refresh: {latest_refresh_text}\n\n"
                f"{_build_macro_snapshot_df(df).to_markdown(index=False)}"
            ),
            metadata={
                "type": "market_environment_snapshot",
                "source": "Macro SQL aggregate",
                "latest_refresh": latest_refresh_text,
            },
        )
    ]

    for category, category_df in df.groupby("category"):
        sections = []
        for _, indicator_df in category_df.groupby("indicator_key"):
            history_df = (
                indicator_df.sort_values("observation_date", ascending=False)
                .head(max_history_per_indicator)
                .copy()
            )
            history_df["observation_date"] = history_df["observation_date"].dt.strftime("%Y-%m-%d")
            history_df["formatted_value"] = history_df.apply(
                lambda row: _format_macro_value(row["value"], row["units"]),
                axis=1,
            )
            display_df = history_df[["observation_date", "formatted_value"]].rename(
                columns={
                    "observation_date": "Observation Date",
                    "formatted_value": "Value",
                }
            )

            sections.append(
                f"## {indicator_df['indicator_name'].iloc[0]}\n"
                f"Release: {indicator_df['release_name'].iloc[0]}\n"
                f"Source: {indicator_df['source'].iloc[0]}\n"
                f"Series ID: {indicator_df['series_id'].iloc[0]}\n"
                f"Notes: {indicator_df['notes'].iloc[0]}\n\n"
                f"{display_df.to_markdown(index=False)}"
            )

        documents.append(
            Document(
                text=(
                    f"**Market Environment - {category.replace('_', ' ').title()}**\n\n"
                    f"Latest refresh: {latest_refresh_text}\n\n"
                    + "\n\n".join(sections)
                ),
                metadata={
                    "type": "market_environment_history",
                    "category": category,
                    "source": "Macro SQL aggregate",
                    "latest_refresh": latest_refresh_text,
                },
            )
        )

    return documents


def _persist_documents_as_index(documents, persist_dir):
    if not documents:
        return None

    ingest_stock.env()
    node_parser = SentenceSplitter(chunk_size=550, chunk_overlap=60)
    nodes = node_parser.get_nodes_from_documents(documents)

    try:
        _reset_persist_dir(persist_dir)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=persist_dir)
        return {"index": index, "persist_dir": persist_dir}
    except PermissionError:
        if os.path.exists(persist_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)
                index.insert_nodes(nodes)
                index.storage_context.persist(persist_dir=persist_dir)
                print(
                    f"Could not rebuild {persist_dir} because the folder is locked; "
                    "appended fresh nodes to the existing index instead."
                )
                return {"index": index, "persist_dir": persist_dir}
            except Exception as exc:
                normalized_persist_dir = os.path.normpath(persist_dir)
                fallback_dir = os.path.join(
                    os.path.dirname(normalized_persist_dir),
                    f"{os.path.basename(normalized_persist_dir)}_refresh",
                )
                if os.path.exists(fallback_dir):
                    shutil.rmtree(fallback_dir, ignore_errors=True)
                index = VectorStoreIndex(nodes)
                index.storage_context.persist(persist_dir=fallback_dir)
                print(
                    f"Could not overwrite locked index at {persist_dir}; "
                    f"wrote the refreshed index to {fallback_dir} instead ({exc})."
                )
                return {"index": index, "persist_dir": fallback_dir}
        raise


def _is_windows_reparse_point(path):
    try:
        path_stat = os.lstat(path)
    except (FileNotFoundError, OSError):
        return False

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and getattr(path_stat, "st_file_attributes", 0) & reparse_flag)


def _reset_persist_dir(persist_dir):
    if not os.path.lexists(persist_dir):
        return

    if _is_windows_reparse_point(persist_dir):
        try:
            os.chmod(persist_dir, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass
        try:
            os.rmdir(persist_dir)
            return
        except OSError:
            pass

    try:
        shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
        return
    except FileNotFoundError:
        return
    except PermissionError:
        if _is_windows_reparse_point(persist_dir):
            os.chmod(persist_dir, stat.S_IWRITE | stat.S_IREAD)
            os.rmdir(persist_dir)
            return
        raise


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        func(path)
    except OSError:
        raise exc_info[1]


def refresh_macro_index(
    db_path=DEFAULT_MACRO_DB_PATH,
    stale_after_hours=24,
    macro_history_per_indicator=12,
    persist_dir=DEFAULT_MACRO_STORAGE_DIR):

    refresh_market_environment_if_stale(db_path=db_path,stale_after_hours=stale_after_hours)
    docs = build_market_environment_docs(
        db_path=db_path,
        max_history_per_indicator=macro_history_per_indicator)
    if not docs:
        print("No market environment documents generated - skipping index update")
        return None

    persist_result = _persist_documents_as_index(docs, persist_dir=persist_dir)
    actual_persist_dir = persist_result["persist_dir"]
    print(f"Market environment index successfully refreshed at {actual_persist_dir}")
    return actual_persist_dir
