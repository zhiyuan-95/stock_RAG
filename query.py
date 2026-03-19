import os

from llama_index.core import Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv

import ingest_stock

load_dotenv("config.env")

DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")
DEFAULT_MACRO_STORAGE_DIR = os.getenv("MACRO_STORAGE_DIR", "./storage/macro")


ANALYSIS_PROMPT = PromptTemplate(
        """You are a disciplined equity research assistant.
        Use only the retrieved database context below. Do not add outside facts, outside valuation data, or assumptions that are not supported by the retrieved text.

        Company: {ticker}
        User question: {question}

        === STOCK DATABASE CONTEXT ===
        {stock_context}

        === MACRO DATABASE CONTEXT ===
        {macro_context}

        The stock context may contain:
        - company overview / business description
        - sector and industry
        - annual and quarterly financial indicators

        The macro context may contain:
        - Fed funds rate
        - CPI / inflation
        - unemployment
        - GDP
        - payrolls
        - PMI and other macro indicators

        Your goal is to extract the most relevant information from those contexts and explain what it means for {ticker}.

        Requirements:
        - Use exact numbers, percentages, and dates from the retrieved context whenever possible.
        - Prioritize information that is actually present in the database context.
        - If something is missing from the context, say that clearly instead of guessing.
        - Do not provide price targets, valuation multiples, or named competitors unless they are explicitly present in the retrieved context.
        - Connect company fundamentals to the macro environment in a concrete way.

        Follow this structure:

        1. Business Overview
        - What the company does
        - Its sector and industry
        - The most important business characteristics mentioned in context

        2. Financial Trend Analysis
        - Revenue trend
        - Margin trend
        - Earnings / EPS trend
        - Cash flow trend
        - Balance sheet / leverage observations
        - 2 to 4 key positives and 2 to 4 key negatives from the financial data

        3. Macro Impact Analysis
        - Which macro indicators matter most for this company
        - Whether the current macro environment is supportive, neutral, or adverse
        - How inflation, rates, growth, labor data, and PMI affect the business if relevant

        4. Risks And Missing Information
        - Main risks visible from the retrieved context
        - Important missing information that limits confidence

        5. Final Judgment
        - Give a concise overall view in 3 to 5 sentences

        Write clearly, quantitatively, and conservatively."""
)

def _load_retriever(persist_dir, similarity_top_k=4):
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(f"Index directory not found: {persist_dir}")

    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)


def _node_text(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)
    if hasattr(node, "get_content"):
        return node.get_content().strip()
    return str(getattr(node, "text", "")).strip()


def _retrieve_context_from_dir(persist_dir, query_str, label, similarity_top_k=4):
    try:
        retriever = _load_retriever(persist_dir, similarity_top_k=similarity_top_k)
    except (FileNotFoundError, ValueError):
        return None

    nodes = retriever.retrieve(query_str)
    chunks = []
    seen = set()

    for node in nodes:
        text = _node_text(node)
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append(text)

    if not chunks:
        return None

    return f"{label}:\n" + "\n\n".join(chunks)


def _resolve_macro_storage_dir(macro_storage_dir):
    candidates = [macro_storage_dir]
    normalized_dir = os.path.normpath(macro_storage_dir)
    fallback_dir = os.path.join(
        os.path.dirname(normalized_dir),
        f"{os.path.basename(normalized_dir)}_refresh",
    )
    candidates.append(fallback_dir)

    for candidate in candidates:
        docstore_path = os.path.join(candidate, "docstore.json")
        if os.path.isdir(candidate) and os.path.isfile(docstore_path):
            return candidate

    return macro_storage_dir


def _ensure_stock_index_directory(ticker, stock_storage_base_dir):
    ticker = ticker.upper()
    persist_dir = os.path.join(stock_storage_base_dir, ticker)
    if os.path.isdir(persist_dir):
        return persist_dir

    print(f"{ticker} is missing from storage/stock; ingesting it now.")
    ingest_stock.refresh_ticker_data_and_index(
        ticker,
        storage_base_dir=stock_storage_base_dir,
    )
    return persist_dir


def get_analysis_context(
    ticker,
    query_str,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    stock_top_k=4,
    macro_top_k=4):
    ingest_stock.env()

    ticker = ticker.upper()
    stock_persist_dir = _ensure_stock_index_directory(
        ticker,
        stock_storage_base_dir=stock_storage_base_dir,
    )
    resolved_macro_storage_dir = _resolve_macro_storage_dir(macro_storage_dir)

    stock_context = _retrieve_context_from_dir(
        stock_persist_dir,
        query_str,
        label=f"{ticker} financial context",
        similarity_top_k=stock_top_k,
    )
    macro_context = _retrieve_context_from_dir(
        resolved_macro_storage_dir,
        query_str,
        label="Macro market context",
        similarity_top_k=macro_top_k,
    )

    return {
        "stock_context": stock_context,
        "macro_context": macro_context,
        "stock_persist_dir": stock_persist_dir,
        "macro_persist_dir": resolved_macro_storage_dir,
    }


def analyze_company(
    ticker,
    custom_query=None,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    stock_top_k=4,
    macro_top_k=4):
    ticker = ticker.upper()
    query_str = custom_query or (
        f"Provide short-term and long-term analysis for {ticker} using both the "
        "company financial information and the current macro market environment."
    )

    context = get_analysis_context(
        ticker=ticker,
        query_str=query_str,
        stock_storage_base_dir=stock_storage_base_dir,
        macro_storage_dir=macro_storage_dir,
        stock_top_k=stock_top_k,
        macro_top_k=macro_top_k,
    )

    if not context["stock_context"] and not context["macro_context"]:
        raise FileNotFoundError(
            "No retrievable stock or macro index was found. Refresh the stock index "
            f"at {context['stock_persist_dir']} and the macro index at "
            f"{context['macro_persist_dir']} first."
        )

    stock_context = context["stock_context"] or "Stock-specific context was not available."
    macro_context = context["macro_context"] or "Macro market context was not available."

    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        question=query_str,
        stock_context=stock_context,
        macro_context=macro_context,
    )

    response = Settings.llm.complete(prompt)
    return str(response)
