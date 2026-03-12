import sqlite3

import os
import yfinance as yf
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.readers.llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_API_KEY'))

def get_financial_summary(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = f"""
    Company: {info.get('longName')}
    Sector: {info.get('sector')} | Industry: {info.get('industry')}
    Market Cap: ${info.get('marketCap'):,} | Trailing P/E: {info.get('trailingPE'):.1f}
    Revenue TTM: ${info.get('totalRevenue'):,} | Net Income TTM: ${info.get('netIncomeToCommon'):,}
    """
    # You can add historical prices, ratios, etc.
    return summary

def build_documents(ticker: str, data_dir = None):
    docs = []
    """
    Build list of Document objects for the given ticker.

    - Always includes financial summary from yfinance
    - If data_dir is provided and exists, also loads & parses files from that directory
    - Later: can add news/transcripts/etc. independently of files
    """
    # 1. Financial snapshot as text document
    fin_text = get_financial_summary(ticker)
    docs.append(Document(text=fin_text, metadata={"ticker": ticker, "type": "financial_summary"}))

    if data_dir:
    # 2. SEC filings via LlamaParse (best for tables in 2026)
        parser = LlamaParse(api_key=os.getenv("LLAMA_PARSE_API_KEY"), result_type="markdown")
        filings = ["latest_10k.pdf", "latest_10q.pdf"]  # download them first or automate
        for file in filings:
            parsed_docs = parser.load_data(f"{data_dir}/{file}")
            for doc in parsed_docs:
                doc.metadata.update({"ticker": ticker, "filing_type": "10-K" if "10k" in file else "10-Q"})
            docs.extend(parsed_docs)
    # 3. Add news / transcripts similarly (use Web readers or downloaded PDFs)
    return docs

# Build / update index
def create_or_update_index(ticker: str):
    docs = build_documents(ticker.upper())
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)
    # Simple in-memory for start -> later use Qdrant/Pinecone/Chroma
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=f"./storage/{ticker}")
    return index

def get_historical_financial_indicators(ticker):
    """
    Fetches all available historical financial statements from yfinance,
    computes key indicators/ratios for each period, and returns as a DataFrame (table).
    """
    stock = yf.Ticker(ticker)

    # Fetch annual statements (use .quarterly_income_stmt etc. for quarterly)
    income = stock.income_stmt.T  # Transpose so rows are dates, columns are items
    balance = stock.balance_sheet.T
    cash_flow = stock.cashflow.T

    # Align on common dates (yfinance may have slight mismatches)
    common_dates = income.index.intersection(balance.index).intersection(cash_flow.index)
    income = income.loc[common_dates]
    balance = balance.loc[common_dates]
    cash_flow = cash_flow.loc[common_dates]

    # Create DataFrame for indicators
    indicators = pd.DataFrame(index=common_dates)

    # Key items from income statement
    indicators['Total Revenue'] = income.get('Total Revenue', pd.NA)
    indicators['Gross Profit'] = income.get('Gross Profit', pd.NA)
    indicators['Operating Income'] = income.get('Operating Income', pd.NA)
    indicators['Net Income'] = income.get('Net Income', pd.NA)
    indicators['Basic EPS'] = income.get('Basic EPS', pd.NA)

    # Key items from balance sheet
    indicators['Total Assets'] = balance.get('Total Assets', pd.NA)
    indicators['Total Liabilities'] = balance.get('Total Liabilities Net Minority Interest', pd.NA)
    indicators['Shareholders Equity'] = balance.get('Common Stock Equity', pd.NA)

    # Key items from cash flow
    indicators['Operating Cash Flow'] = cash_flow.get('Operating Cash Flow', pd.NA)
    indicators['Free Cash Flow'] = cash_flow.get('Free Cash Flow', pd.NA)

    # Computed ratios (add more as needed for your analysis)
    indicators['Gross Margin'] = indicators['Gross Profit'] / indicators['Total Revenue']
    indicators['Operating Margin'] = indicators['Operating Income'] / indicators['Total Revenue']
    indicators['Net Margin'] = indicators['Net Income'] / indicators['Total Revenue']
    indicators['ROE'] = indicators['Net Income'] / indicators['Shareholders Equity']
    indicators['ROA'] = indicators['Net Income'] / indicators['Total Assets']
    indicators['Debt to Equity'] = indicators['Total Liabilities'] / indicators['Shareholders Equity']

    # Finalize table
    indicators = indicators.reset_index().rename(columns={'index': 'Date'})
    indicators['Ticker'] = ticker

    return indicators

# Example integration into build_documents (modify as per your existing setup)
def build_documents(tickers, data_dir=None):
    documents = []
    for ticker in tickers:
        # Fetch historical indicators table
        indicators_df = get_historical_financial_indicators(ticker)
        # Convert to Markdown to preserve table structure for RAG embedding
        indicators_md = indicators_df.to_markdown(index=False)
        # Create Document for ingestion into vector store
        doc = Document(
            text=f"Historical financial indicators for {ticker}:\n\n{indicators_md}",
            metadata={"ticker": ticker, "type": "financial_indicators", "source": "yfinance"}
        )
        documents.append(doc)
        # Optional: Persist raw table to SQLite for structured storage/queries
        # (e.g., for hybrid RAG with SQL tools)
        import sqlite3
        conn = sqlite3.connect('stock_data.db')  # Your storage DB path
        indicators_df.to_sql('financial_indicators', conn, if_exists='append', index=False)
        conn.close()

    # Add other sources (e.g., news, competitors) here as before

    return documents

print(get_historical_financial_indicators('aapl'))
# Usage example (run this to ingest)
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('config.env')

    # Your tickers list, e.g., from prior Nasdaq/Dow fetch
    tickers = ['AAPL', 'MSFT']  # Replace with your list
    docs = build_documents(tickers)
    # Then build/index your vector store as before, e.g.:
    # from llama_index.core import VectorStoreIndex
    # index = VectorStoreIndex.from_documents(docs)
    # index.storage_context.persist(persist_dir="./storage")
