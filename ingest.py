import sqlite3
import yfinance as yf
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from datetime import datetime
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv('config.env')

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_API_KEY'))


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


# Example integration into build_documents (modify as per your existing setup)
def build_financial_docs(tickers, max_quarters=12):
    documents = []

    for ticker in tickers:
        # Get last 12 quarters
        df_q = get_historical_financial_indicators(ticker, frequency='quarterly', max_periods=max_quarters)

        if df_q.empty:
            continue

        md_table = df_q.to_markdown(index=False)   # assuming tabulate is installed

        text = (
            f"**Last {len(df_q)} Quarterly Financial Indicators — {ticker}**\n\n"
            f"Most recent quarter: {df_q['Period End'].iloc[0]}\n"
            f"Oldest in this set: {df_q['Period End'].iloc[-1]}\n\n"
            f"{md_table}"
        )

        doc = Document(
            text=text,
            metadata={
                "ticker": ticker.upper(),
                "type": "financial_indicators",
                "frequency": "Quarterly",
                "periods": len(df_q),
                "most_recent": df_q['Period End'].iloc[0],
                "source": "yfinance"
            }
        )
        documents.append(doc)

    # You can also add annual version the same way if desired
    return documents

def create_or_update_index(ticker: str):
    docs = build_financial_docs(ticker.upper())  # your function

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)

    persist_dir = f"./storage/{ticker}"

    try:
        # Try to load existing
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        # If exists → insert new nodes
        index.insert_nodes(nodes)
    except:
        # First time → create new
        index = VectorStoreIndex(nodes)

    index.storage_context.persist(persist_dir=persist_dir)
    return index
#print(get_historical_financial_indicators('NVDA'))
#print(build_financial_docs(['NVDA']))
#print(create_or_update_index(['NVDA']))
# New function to update records in SQLite storage
def update_financial_records(ticker, db_path='stock_data.db'):
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
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['Period End Date'], keep='last')  # keep newest if conflict

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


if __name__ == "__main__":
    # Assume your DB is set up with the table (create if not)
    conn = sqlite3.connect('stock_data.db')
    # Create table if not exists (adjust columns to match your df)
    create_table_query = """
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
    conn.execute(create_table_query)
    conn.close()

    # Update for a ticker
    update_financial_records('NVDA')
