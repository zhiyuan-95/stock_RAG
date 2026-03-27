import os
import re
import sqlite3

import pandas as pd
from llama_index.core import Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv

import ingest_graph
import ingest_knowledge
import ingest_stock

load_dotenv("config.env")

DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")
DEFAULT_KNOWLEDGE_STORAGE_DIR = os.getenv("KNOWLEDGE_STORAGE_DIR", "./storage/knowledge")
DEFAULT_MACRO_STORAGE_DIR = os.getenv("MACRO_STORAGE_DIR", "./storage/macro")
DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")


ANALYSIS_PROMPT = PromptTemplate(
        """You are a disciplined equity research assistant.
        Use only the retrieved database context below. Do not add outside facts, outside valuation data, or assumptions that are not supported by the retrieved text.

        Company: {ticker}
        User question: {question}

        === STOCK DATABASE CONTEXT ===
        {stock_context}

        === KNOWLEDGE DATABASE CONTEXT ===
        {knowledge_context}

        === MACRO DATABASE CONTEXT ===
        {macro_context}

        === GRAPH LAYER CONTEXT ===
        {graph_context}

        The stock context may contain:
        - company overview / business description
        - sector and industry
        - SEC 10-K / 10-Q filing sections
        - Item 1 / Item 1A / MD&A / financial statements
        - annual and quarterly financial indicators

        The knowledge context may contain:
        - glossary definitions for common financial indicators
        - plain-English explanation of what a metric means
        - common reasons a metric improves or deteriorates
        - general examples that are not specific to {ticker}

        The macro context may contain:
        - Fed funds rate
        - CPI / inflation
        - unemployment
        - GDP
        - payrolls
        - PMI and other macro indicators

        The graph context may contain:
        - glossary indicator concepts and their group/subgroup relationships
        - company filing docs connected to the company node
        - latest stock and macro observations connected to glossary indicators
        - one-hop relationships that connect definitions, filings, and observations

        Your goal is to extract the most relevant information from those contexts and explain what it means for {ticker}.

        Requirements:
        - Use exact numbers, percentages, and dates from the retrieved context whenever possible.
        - When multiple filing periods or years are present, prioritize the most recent dated information unless the user explicitly asks for history.
        - For current financial trend analysis, prioritize the latest annual and quarterly financial indicator docs over older filing financial statements.
        - Prioritize information that is actually present in the database context.
        - If something is missing from the context, say that clearly instead of guessing.
        - Do not provide price targets, valuation multiples, or named competitors unless they are explicitly present in the retrieved context.
        - Connect company fundamentals to the macro environment in a concrete way.
        - Use SEC filing sections to explain the business, risks, and the reasons behind financial trends when available.
        - If glossary context is retrieved, use it only to explain what a metric means in plain English. Do not treat glossary examples as company-specific facts about {ticker}.

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

        4. Filing Takeaways
        - The most important signals from Item 1 / Item 1A / MD&A / financial statements
        - What the filings say about business drivers, risks, and causality
        - If the filings do not contain enough evidence for the question, say that directly

        5. Risks And Missing Information
        - Main risks visible from the retrieved context
        - Important missing information that limits confidence

        6. Final Judgment
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


def _retrieve_chunks_from_dir(persist_dir, query_str, similarity_top_k=4):
    try:
        retriever = _load_retriever(persist_dir, similarity_top_k=similarity_top_k)
    except (FileNotFoundError, ValueError):
        return []

    nodes = retriever.retrieve(query_str)
    chunks = []
    seen = set()

    for node in nodes:
        text = _node_text(node)
        if not text or text in seen:
            continue
        seen.add(text)
        chunks.append(text)

    return chunks


def _format_context_chunks(label, chunks):
    if not chunks:
        return None
    return f"{label}:\n" + "\n\n".join(chunks)


def _limit_context(text, max_chars):
    if not text:
        return text
    normalized_text = str(text).strip()
    if len(normalized_text) <= max_chars:
        return normalized_text
    return normalized_text[: max_chars - 3].rstrip() + "..."


def _retrieve_context_from_dir(persist_dir, query_str, label, similarity_top_k=4):
    chunks = _retrieve_chunks_from_dir(
        persist_dir,
        query_str,
        similarity_top_k=similarity_top_k,
    )
    return _format_context_chunks(label, chunks)


def _direct_financial_indicator_context(ticker, db_path=ingest_stock.DEFAULT_STOCK_DB_PATH):
    if not db_path or not os.path.isfile(db_path):
        return None

    selected_columns = [
        "Period End Date",
        "Total Revenue",
        "Gross Margin",
        "Operating Margin",
        "Net Profit Margin",
        "Diluted EPS",
        "Free Cash Flow",
        "Current Ratio",
        "Debt-to-Equity (D/E)",
        "Return on Equity (ROE)",
        "Price-to-Earnings (P/E) Trailing",
        "Current Market Price",
    ]

    conn = sqlite3.connect(db_path)
    sections = []
    try:
        for frequency, limit in (("Quarterly", 4), ("Annual", 4)):
            df = pd.read_sql_query(
                """
                SELECT *
                FROM financial_indicators
                WHERE Ticker = ? AND Frequency = ?
                ORDER BY date([Period End Date]) DESC
                LIMIT ?
                """,
                conn,
                params=(ticker.upper(), frequency, limit),
            )
            if df.empty:
                continue

            keep_columns = [column for column in selected_columns if column in df.columns]
            if not keep_columns:
                continue

            display_df = df[keep_columns].copy()
            sections.append(
                "\n".join(
                    [
                        f"Latest {len(display_df)} {frequency} financial indicator records from SQL",
                        display_df.to_markdown(index=False),
                    ]
                )
            )
    finally:
        conn.close()

    if not sections:
        return None

    return "Structured stock indicator context:\n" + "\n\n".join(sections)


def _matched_knowledge_chunks(query_str, max_matches=3):
    normalized_query = re.sub(r"\s+", " ", query_str.lower()).strip()
    exact_matches = []

    for doc in ingest_knowledge.build_glossary_docs():
        candidate_terms = []
        for key in ("indicator_name", "indicator_canonical_name", "group", "subgroup"):
            value = (doc.metadata.get(key) or "").strip()
            if value:
                candidate_terms.append(value)
        candidate_terms.extend(doc.metadata.get("indicator_aliases", []))

        best_match_length = 0
        for term in candidate_terms:
            normalized_term = re.sub(r"\s+", " ", str(term).lower()).strip()
            if normalized_term and normalized_term in normalized_query:
                best_match_length = max(best_match_length, len(normalized_term))

        if best_match_length:
            exact_matches.append((best_match_length, doc.text))

    exact_matches.sort(key=lambda item: item[0], reverse=True)

    matched_chunks = []
    seen = set()
    for _, text in exact_matches:
        if text in seen:
            continue
        seen.add(text)
        matched_chunks.append(text)
        if len(matched_chunks) >= max_matches:
            break

    return matched_chunks


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


def _ensure_knowledge_index_directory(knowledge_storage_dir):
    docstore_path = os.path.join(knowledge_storage_dir, "docstore.json")
    if os.path.isdir(knowledge_storage_dir) and os.path.isfile(docstore_path):
        return knowledge_storage_dir

    print("Knowledge index is missing; ingesting glossary knowledge now.")
    ingest_knowledge.refresh_knowledge_index(storage_dir=knowledge_storage_dir)
    return knowledge_storage_dir


def _ensure_graph_index_directory(
    ticker,
    graph_storage_dir,
):
    if ingest_graph.graph_index_exists(ticker=ticker, storage_dir=graph_storage_dir):
        return os.path.join(graph_storage_dir, ticker.upper())

    print("Graph layer is missing; building the graph index now.")
    try:
        persist_dir = ingest_graph.refresh_property_graph_for_ticker(
            ticker,
            stock_db_path=ingest_stock.DEFAULT_STOCK_DB_PATH,
            macro_db_path=os.getenv("MACRO_SQL_DB_PATH", "macro_data.db"),
            filings_base_dir=ingest_stock.DEFAULT_STOCK_FILINGS_BASE_DIR,
            glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
            metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
            storage_dir=graph_storage_dir,
        )
    except Exception as exc:
        print(f"Graph layer refresh skipped for {ticker}: {exc}")
        persist_dir = None
    return persist_dir or os.path.join(graph_storage_dir, ticker.upper())


def get_analysis_context(
    ticker,
    query_str,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    knowledge_storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    stock_top_k=4,
    knowledge_top_k=3,
    macro_top_k=4,
    graph_top_k=4):
    ingest_stock.env()

    ticker = ticker.upper()
    stock_persist_dir = _ensure_stock_index_directory(
        ticker,
        stock_storage_base_dir=stock_storage_base_dir,
    )
    resolved_knowledge_storage_dir = _ensure_knowledge_index_directory(knowledge_storage_dir)
    resolved_macro_storage_dir = _resolve_macro_storage_dir(macro_storage_dir)
    resolved_graph_storage_dir = _ensure_graph_index_directory(
        ticker,
        graph_storage_dir=graph_storage_dir,
    )

    stock_chunks = []
    seen_stock_chunks = set()
    financial_doc_markers = (
        "Latest 12 Quarterly Financial Indicators",
        "Latest 10 Annual Financial Indicators",
        "Current market price used for market-based ratios",
    )
    exact_indicator_queries = [
        f"{ticker} Latest Quarterly Financial Indicators revenue margins eps cash flow current market price",
        f"{ticker} Latest Annual Financial Indicators revenue margins eps cash flow balance sheet leverage",
    ]
    for stock_query in exact_indicator_queries:
        for chunk in _retrieve_chunks_from_dir(
            stock_persist_dir,
            stock_query,
            similarity_top_k=max(stock_top_k, 12),
        ):
            if not any(marker in chunk for marker in financial_doc_markers):
                continue
            if chunk in seen_stock_chunks:
                continue
            seen_stock_chunks.add(chunk)
            stock_chunks.append(chunk)
            if len(stock_chunks) >= 2:
                break
        if len(stock_chunks) >= 2:
            break

    stock_query_plan = [
        (
            f"{ticker} company overview business description sector industry",
            1,
        ),
        (
            f"{ticker} sec item 1 business item 1a risk factors item 7 md&a business drivers risks",
            2,
        ),
        (
            query_str,
            2,
        ),
    ]
    for stock_query, chunk_budget in stock_query_plan:
        added_for_query = 0
        for chunk in _retrieve_chunks_from_dir(
            stock_persist_dir,
            stock_query,
            similarity_top_k=stock_top_k,
        ):
            if chunk in seen_stock_chunks:
                continue
            seen_stock_chunks.add(chunk)
            stock_chunks.append(chunk)
            added_for_query += 1
            if added_for_query >= chunk_budget or len(stock_chunks) >= 8:
                break
        if len(stock_chunks) >= 8:
            break

    stock_context = _format_context_chunks(
        f"{ticker} financial context",
        stock_chunks,
    )
    direct_financial_context = _direct_financial_indicator_context(ticker)
    if direct_financial_context:
        stock_context = (
            f"{direct_financial_context}\n\n{stock_context}"
            if stock_context
            else direct_financial_context
        )
    knowledge_chunks = []
    seen_knowledge_chunks = set()
    for chunk in _matched_knowledge_chunks(query_str):
        if chunk in seen_knowledge_chunks:
            continue
        seen_knowledge_chunks.add(chunk)
        knowledge_chunks.append(chunk)

    for chunk in _retrieve_chunks_from_dir(
        resolved_knowledge_storage_dir,
        query_str,
        similarity_top_k=knowledge_top_k,
    ):
        if chunk in seen_knowledge_chunks:
            continue
        seen_knowledge_chunks.add(chunk)
        knowledge_chunks.append(chunk)
        if len(knowledge_chunks) >= max(knowledge_top_k, 3):
            break

    knowledge_context = _format_context_chunks(
        "Financial glossary context",
        knowledge_chunks,
    )
    macro_context = _retrieve_context_from_dir(
        resolved_macro_storage_dir,
        query_str,
        label="Macro market context",
        similarity_top_k=macro_top_k,
    )
    graph_context = ingest_graph.retrieve_graph_context(
        query_str,
        ticker=ticker,
        storage_dir=graph_storage_dir,
        similarity_top_k=graph_top_k,
    )
    stock_context = _limit_context(stock_context, 18000)
    knowledge_context = _limit_context(knowledge_context, 5000)
    macro_context = _limit_context(macro_context, 5000)
    graph_context = _limit_context(graph_context, 2500)

    return {
        "stock_context": stock_context,
        "knowledge_context": knowledge_context,
        "macro_context": macro_context,
        "graph_context": graph_context,
        "stock_persist_dir": stock_persist_dir,
        "knowledge_persist_dir": resolved_knowledge_storage_dir,
        "macro_persist_dir": resolved_macro_storage_dir,
        "graph_persist_dir": resolved_graph_storage_dir,
    }


def analyze_company(
    ticker,
    custom_query=None,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    knowledge_storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    stock_top_k=4,
    knowledge_top_k=3,
    macro_top_k=4,
    graph_top_k=4):
    ticker = ticker.upper()
    query_str = custom_query or (
        f"Provide short-term and long-term analysis for {ticker} using both the "
        "company financial information and the current macro market environment."
    )

    context = get_analysis_context(
        ticker=ticker,
        query_str=query_str,
        stock_storage_base_dir=stock_storage_base_dir,
        knowledge_storage_dir=knowledge_storage_dir,
        macro_storage_dir=macro_storage_dir,
        graph_storage_dir=graph_storage_dir,
        stock_top_k=stock_top_k,
        knowledge_top_k=knowledge_top_k,
        macro_top_k=macro_top_k,
        graph_top_k=graph_top_k,
    )

    if (
        not context["stock_context"]
        and not context["knowledge_context"]
        and not context["macro_context"]
        and not context["graph_context"]
    ):
        raise FileNotFoundError(
            "No retrievable stock, knowledge, macro, or graph index was found. Refresh the stock index "
            f"at {context['stock_persist_dir']}, the knowledge index at {context['knowledge_persist_dir']}, and the macro index at "
            f"{context['macro_persist_dir']}, and the graph index at {context['graph_persist_dir']} first."
        )

    stock_context = context["stock_context"] or "Stock-specific context was not available."
    knowledge_context = context["knowledge_context"] or "Financial glossary context was not available."
    macro_context = context["macro_context"] or "Macro market context was not available."
    graph_context = context["graph_context"] or "Graph layer context was not available."

    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        question=query_str,
        stock_context=stock_context,
        knowledge_context=knowledge_context,
        macro_context=macro_context,
        graph_context=graph_context,
    )

    response = Settings.llm.complete(prompt)
    return str(response)
