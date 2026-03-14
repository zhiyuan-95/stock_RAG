# query.py
from typing import Optional
from llama_index.core import (
    Settings,
    load_index_from_storage,
    StorageContext,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from dotenv import load_dotenv
import os

load_dotenv('config.env')

# Reuse the same global settings as ingest.py (you can move to a config.py later)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key = os.getenv('OPENAI_API_KEY'))

# I am not adding this right now, since I don't have the function that analysis new
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
INVESTMENT_ADVICE_PROMPT = PromptTemplate(
    """
    You are a senior equity research analyst. Provide balanced, factual, data-grounded short-term and long-term investment advice for {ticker}, focusing on potential opportunities, risks, and catalysts.

    Use ONLY the provided context (financial indicators, reports, news, customers, competitors). Do NOT make up information or speculate beyond the data. This is not personalized financial advice—advise consulting professionals.

    Structure your answer exactly like this:

    1. Key Financial Trends
       - Recent quarterly indicators (e.g., revenue growth, margins, ROE/ROA from last 4–12 quarters)
       - Multi-year annual trends (e.g., 3–8 year patterns in debt, cash flow, EPS)

    2. Market Positioning
       - Major customers, suppliers, partners, and concentration risks
       - Key competitors, strengths/weaknesses, and relative performance

    3. Recent News & External Factors
       - Highlights from latest reports/news and their implications

    4. Short-Term Advice (1–6 months)
       - Tactical outlook: Based on recent quarters, earnings momentum, news events (e.g., product launches, macro factors)
       - Potential catalysts/risks: e.g., upcoming earnings, supply chain issues
       - Advice: Hold/Buy/Sell rationale, with probability estimates if data supports

    5. Long-Term Advice (3–10+ years)
       - Strategic outlook: Based on multi-year trends, competitive moat, growth drivers (e.g., market expansion, innovation)
       - Potential catalysts/risks: e.g., industry shifts, regulatory changes
       - Advice: Hold/Buy/Sell rationale, with valuation considerations (e.g., compared to historical averages)

    6. Overall Recommendation
       - Balanced summary with key things to watch

    Cite sources (e.g., quarter/year from indicators, report date, news date) inline.

    Context information:
    ---------------------
    {context_str}
    ---------------------

    Query: {query_str}
    Answer in clear, professional language:
    """
)

# In get_analysis_engine or analyze_company
response = engine.query(
    INVESTMENT_ADVICE_PROMPT.format(ticker=ticker.upper(), query_str=query_str, context_str=context_str)
)

def get_analysis_engine(
        ticker: str,
        similarity_top_k: int = 2,
#        similarity_cutoff: float = 0.6,
        storage_base_dir: str = "./storage",
    ):
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
#    print(retriever.retrieve("NVDA financials"))
#    prompt = ANALYSIS_PROMPT.partial_format(ticker=ticker.upper())
    response_synthesizer = get_response_synthesizer()

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def analyze_company(ticker: str, custom_query: Optional[str] = None) -> str:
    """Main function: get full analysis or answer custom question"""
    engine = get_analysis_engine(ticker)

    if custom_query:
        response = engine.query(custom_query)
    else:
        # Default full company analysis
        response = engine.query(
            INVESTMENT_ADVICE_PROMPT.format(ticker=ticker.upper(), query_str=query_str, context_str=context_str)
            )
    return str(response)
