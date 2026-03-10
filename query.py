# query.py
from typing import Optional
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    PromptTemplate,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from dotenv import load_dotenv
import os

load_dotenv()

# Reuse the same global settings as ingest.py (you can move to a config.py later)
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


ANALYSIS_PROMPT = PromptTemplate(
        """
            You are a senior equity research analyst. Provide a concise, factual, data-grounded analysis of {ticker}.

            Use ONLY the provided context. Do NOT make up information.

            Structure your answer exactly like this:

            1. Company Overview & Key Financials
               - Current metrics (market cap, P/E, revenue TTM, etc.)
               - Recent trends

            2. Highlights from Latest Filings
               - Revenue drivers, risks, strategy

            3. Recent News & Sentiment
               - Key events and potential impact

            4. Customers / Suppliers / Partners
               - Major mentioned ones + concentration risks

            5. Competitors & Market Positioning
               - Key rivals + strengths/weaknesses

            6. Overall Considerations
               - Strengths, weaknesses, things to watch

            Cite sources (filing type/date, news date, etc.) when possible.

            Context information:
            ---------------------
            {{context_str}}
            ---------------------

            Query: {{query_str}}
            Answer in clear, professional language:
        """
    )


def get_analysis_engine(
        ticker: str,
        similarity_top_k: int = 8,
        similarity_cutoff: float = 0.78,
        storage_base_dir: str = "./storage",
    ) -> RetrieverQueryEngine:
    """Load persisted index and create query engine for analysis"""
    persist_dir = os.path.join(storage_base_dir, ticker.upper())

    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"No stored index found for {ticker} at {persist_dir}\n"
            "Run ingest.py first to build the index."
        )

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        text_qa_template=ANALYSIS_PROMPT,
    )


def analyze_company(ticker: str, custom_query: Optional[str] = None) -> str:
    """Main function: get full analysis or answer custom question"""
    engine = get_analysis_engine(ticker)

    if custom_query:
        response = engine.query(custom_query)
    else:
        # Default full company analysis
        response = engine.query(
            f"Provide a complete fundamental analysis of {ticker}"
        )

    return str(response)


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    custom_q = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    print(f"\nAnalyzing {ticker}...\n")
    result = analyze_company(ticker, custom_q)
    print(result)
