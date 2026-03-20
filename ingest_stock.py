import sqlite3
import yfinance as yf
from llama_index.core import (Settings, Document, VectorStoreIndex, StorageContext,load_index_from_storage)
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv("config.env")

DEFAULT_STOCK_DB_PATH = os.getenv("STOCK_SQL_DB_PATH", "stock_data.db")
DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")

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


def get_company_profile(ticker):
    ticker = ticker.upper()
    stock = yf.Ticker(ticker)

    try:
        info = stock.get_info()
    except Exception as exc:
        print(f"Skipping company profile fetch for {ticker}: {exc}")
        info = {}

    company_name = (
        _clean_profile_field(info.get("longName"))
        or _clean_profile_field(info.get("shortName"))
        or ticker
    )
    sector = _clean_profile_field(info.get("sector"))
    industry = _clean_profile_field(info.get("industry"))
    description = (
        _clean_profile_field(info.get("longBusinessSummary"))
        or _clean_profile_field(info.get("description"))
    )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "description": description,
        "source": "Yahoo Finance via yfinance",
    }


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
def build_financial_docs(ticker, db_path=DEFAULT_STOCK_DB_PATH, max_quarters=12, max_annual=8):
    documents = []
    conn = sqlite3.connect(db_path)

    ticker = ticker.upper()
    company_profile = get_company_profile(ticker)

    profile_lines = [
        f"Company: {company_profile['company_name']}",
        f"Ticker: {ticker}",
        f"Sector: {company_profile['sector'] or 'Unknown'}",
        f"Industry: {company_profile['industry'] or 'Unknown'}",
    ]
    if company_profile["description"]:
        profile_lines.append("")
        profile_lines.append("Business Description:")
        profile_lines.append(company_profile["description"])

    documents.append(
        Document(
            text="**Company Overview**\n\n" + "\n".join(profile_lines),
            metadata={
                "ticker": ticker,
                "type": "company_profile",
                "company_name": company_profile["company_name"],
                "sector": company_profile["sector"] or "Unknown",
                "industry": company_profile["industry"] or "Unknown",
                "source": company_profile["source"],
            }
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
                "table_rows": len(df)
            }
        )
        documents.append(doc)

    conn.close()
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

    # Step 2: build documents from the fresh DB
    docs = build_financial_docs(ticker, db_path=db_path,max_quarters=max_quarters, max_annual=max_annual)

    if not docs:
        print(f"No documents generated for {ticker} — skipping index update")
        return None

    # Step 3: upsert into vector store
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)

    os.makedirs(storage_base_dir, exist_ok=True)
    persist_dir = os.path.join(storage_base_dir, ticker)

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
