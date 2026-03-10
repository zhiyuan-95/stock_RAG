import os
import yfinance as yf
from dotenv import load_dotenv
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
#from llama_index.readers.llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
load_dotenv('config.env')

Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_APIKEY'))

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
        parser = LlamaParse(api_key=os.getenv("LLAMA_PARSE_APIKEY"), result_type="markdown")
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
    docs = build_documents(ticker)
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)
    # Simple in-memory for start -> later use Qdrant/Pinecone/Chroma
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=f"./storage/{ticker}")
    return index
create_or_update_index('NVDA')
