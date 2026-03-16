import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from io import StringIO

import ingest_stock
import pandas as pd
import requests
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter


FINANCIAL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS financial_indicators (
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
    retrieved_at TEXT
)
"""


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}


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
        "indicator_key": "ism_manufacturing_pmi",
        "indicator_name": "ISM Manufacturing PMI",
        "category": "business_activity",
        "release_name": "ISM Manufacturing PMI Report",
        "source": "Institute for Supply Management",
        "frequency": "Monthly",
        "units": "Percent",
        "notes": "Manufacturing PMI from the official ISM monthly report page.",
        "sector": "manufacturing",
        "history_limit": 12,
    },
    {
        "indicator_key": "ism_services_pmi",
        "indicator_name": "ISM Services PMI",
        "category": "business_activity",
        "release_name": "ISM Services PMI Report",
        "source": "Institute for Supply Management",
        "frequency": "Monthly",
        "units": "Percent",
        "notes": "Services PMI from the official ISM monthly report page.",
        "sector": "services",
        "history_limit": 12,
    },
]


def _ensure_tables(conn):
    conn.execute(FINANCIAL_TABLE_SQL)
    conn.execute(MACRO_TABLE_SQL)


def ensure_sql_tables(db_path="stock_data.db"):
    conn = sqlite3.connect(db_path)
    _ensure_tables(conn)
    conn.commit()
    conn.close()


def _fetch_fred_series_history(series_id, history_limit=None, timeout=20):
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        headers=REQUEST_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    raw_df = pd.read_csv(StringIO(response.text))
    if "DATE" not in raw_df.columns or len(raw_df.columns) < 2:
        raise ValueError(f"Unexpected response while fetching series {series_id}")

    value_column = next(column for column in raw_df.columns if column != "DATE")
    df = raw_df.rename(columns={"DATE": "observation_date", value_column: "value"})
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["observation_date", "value"])
    df = df.sort_values("observation_date", ascending=False).reset_index(drop=True)

    if history_limit is not None:
        df = df.head(history_limit)
    return df


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


def _fetch_ism_pmi_history(series_def, now=None, timeout=20):
    best_df = pd.DataFrame()
    best_url = None

    for url in _ism_report_urls(series_def["sector"], now=now):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
        except Exception:
            continue

        try:
            candidate_df = _extract_ism_history_from_html(
                response.text,
                history_limit=series_def.get("history_limit"),
            )
        except Exception:
            continue
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


def market_environment_needs_refresh(db_path="stock_data.db", stale_after_hours=24):
    ensure_sql_tables(db_path)
    conn = sqlite3.connect(db_path)
    latest_df = pd.read_sql_query(
        "SELECT MAX(retrieved_at) AS latest_retrieved_at FROM macro_indicators",
        conn,
    )
    conn.close()

    latest_value = latest_df["latest_retrieved_at"].iloc[0]
    if pd.isna(latest_value) or not latest_value:
        return True

    latest_dt = pd.to_datetime(latest_value, errors="coerce")
    if pd.isna(latest_dt):
        return True

    return datetime.now() - latest_dt.to_pydatetime() >= timedelta(hours=stale_after_hours)


def update_market_environment_records(db_path="stock_data.db"):
    """
    Fetches the latest macro-market series, stores them in SQLite, and keeps
    each indicator series fresh for downstream document generation.
    """
    ensure_sql_tables(db_path)
    conn = sqlite3.connect(db_path)
    total_rows_written = 0

    for series_def in FRED_SERIES_CATALOG:
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
        base_rows = _rows_from_dataframe(base_df, base_def, retrieved_at)
        indicator_frames = [base_rows]
        indicator_keys_to_replace = [base_def["indicator_key"]]

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

        conn.executemany(
            "DELETE FROM macro_indicators WHERE indicator_key = ?",
            [(indicator_key,) for indicator_key in indicator_keys_to_replace],
        )

        for frame in indicator_frames:
            frame.to_sql("macro_indicators", conn, if_exists="append", index=False)
            total_rows_written += len(frame)

    for series_def in ISM_SERIES_CATALOG:
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


def refresh_market_environment_if_stale(db_path="stock_data.db", stale_after_hours=24):
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


def build_market_environment_docs(db_path="stock_data.db", max_history_per_indicator=12):
    ensure_sql_tables(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT *
        FROM macro_indicators
        ORDER BY category, indicator_name, observation_date DESC
        """,
        conn,
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


def build_company_and_market_docs(
    ticker,
    db_path="stock_data.db",
    max_quarters=12,
    max_annual=8,
    macro_history_per_indicator=12,
):
    ensure_sql_tables(db_path)
    financial_docs = ingest_stock.build_financial_docs(
        ticker,
        db_path=db_path,
        max_quarters=max_quarters,
        max_annual=max_annual,
    )
    market_docs = build_market_environment_docs(
        db_path=db_path,
        max_history_per_indicator=macro_history_per_indicator,
    )

    documents = []
    if financial_docs or market_docs:
        combined_sections = []
        if financial_docs:
            combined_sections.extend(doc.text for doc in financial_docs[:2])
        if market_docs:
            combined_sections.append(market_docs[0].text)

        documents.append(
            Document(
                text=(
                    f"**Combined Company And Market Context - {ticker.upper()}**\n\n"
                    + "\n\n".join(combined_sections)
                ),
                metadata={
                    "ticker": ticker.upper(),
                    "type": "combined_company_market_context",
                    "source": "SQLite aggregate",
                },
            )
        )

    documents.extend(financial_docs)
    documents.extend(market_docs)
    return documents


def build_company_and_market_context(
    ticker,
    db_path="stock_data.db",
    max_quarters=12,
    max_annual=8,
    macro_history_per_indicator=12,
):
    docs = build_company_and_market_docs(
        ticker=ticker,
        db_path=db_path,
        max_quarters=max_quarters,
        max_annual=max_annual,
        macro_history_per_indicator=macro_history_per_indicator,
    )
    return "\n\n".join(doc.text for doc in docs)


def _persist_documents_as_index(documents, persist_dir):
    if not documents:
        return None

    ingest_stock.env()
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def refresh_market_environment_index(
    db_path="stock_data.db",
    stale_after_hours=24,
    macro_history_per_indicator=12,
    persist_dir="./storage/MARKET_ENVIRONMENT",
):
    refresh_market_environment_if_stale(
        db_path=db_path,
        stale_after_hours=stale_after_hours,
    )
    docs = build_market_environment_docs(
        db_path=db_path,
        max_history_per_indicator=macro_history_per_indicator,
    )

    if not docs:
        print("No market environment documents generated - skipping index update")
        return None

    _persist_documents_as_index(docs, persist_dir=persist_dir)
    print(f"Market environment index successfully refreshed at {persist_dir}")
    return persist_dir


def refresh_ticker_macro_index(
    ticker,
    db_path="stock_data.db",
    max_quarters=12,
    max_annual=8,
    refresh_market_environment=True,
    stale_after_hours=24,
    macro_history_per_indicator=12,
):
    """
    Rebuilds one ticker index so it contains both company financial data and
    current macro-market environment context.
    """
    ticker = ticker.upper()
    ensure_sql_tables(db_path)
    ingest_stock.update_financial_records(ticker, db_path=db_path)

    if refresh_market_environment:
        refresh_market_environment_if_stale(
            db_path=db_path,
            stale_after_hours=stale_after_hours,
        )

    docs = build_company_and_market_docs(
        ticker=ticker,
        db_path=db_path,
        max_quarters=max_quarters,
        max_annual=max_annual,
        macro_history_per_indicator=macro_history_per_indicator,
    )
    if not docs:
        print(f"No documents generated for {ticker} - skipping index update")
        return None

    persist_dir = f"./storage/{ticker}"
    _persist_documents_as_index(docs, persist_dir=persist_dir)
    print(f"Ticker + macro index successfully refreshed for {ticker}")
    return persist_dir
