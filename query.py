import json
import os
import re
import sqlite3

import pandas as pd
from llama_index.core import Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv

import ingest_graph
import ingest_knowledge
import ingest_macro
import ingest_stock
load_dotenv("config.env")

DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")
DEFAULT_KNOWLEDGE_STORAGE_DIR = os.getenv("KNOWLEDGE_STORAGE_DIR", "./storage/knowledge")
DEFAULT_MACRO_STORAGE_DIR = os.getenv("MACRO_STORAGE_DIR", "./storage/macro")
DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")
_INDEX_CACHE = {}
_GLOSSARY_DOCS_CACHE = {}
EXPECTED_EMBEDDING_DIM = getattr(ingest_stock, "DEFAULT_EMBED_DIMENSION", 1024)


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
        - active-tier and archive-tier SEC 10-K / 10-Q filing sections
        - short summaries for major filing sections
        - Item 1 / Item 1A / MD&A narrative sections
        - structured filing-linked financial statement facts, indicators, and note summaries
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

    signature = _persist_dir_signature(
        persist_dir,
        ["docstore.json", "index_store.json", "default__vector_store.json"],
    )
    cache_entry = _INDEX_CACHE.get(persist_dir)
    if not cache_entry or cache_entry["signature"] != signature:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        cache_entry = {
            "signature": signature,
            "index": index,
            "retrievers": {},
        }
        _INDEX_CACHE[persist_dir] = cache_entry

    retriever = cache_entry["retrievers"].get(similarity_top_k)
    if retriever is None:
        retriever = VectorIndexRetriever(
            index=cache_entry["index"],
            similarity_top_k=similarity_top_k,
        )
        cache_entry["retrievers"][similarity_top_k] = retriever
    return retriever


def _persist_dir_signature(persist_dir, required_files):
    signature = []
    for file_name in required_files:
        file_path = os.path.join(persist_dir, file_name)
        if not os.path.isfile(file_path):
            signature.append((file_name, None, None))
            continue
        stat_result = os.stat(file_path)
        signature.append((file_name, stat_result.st_mtime_ns, stat_result.st_size))
    return tuple(signature)


def _vector_store_embedding_dim(persist_dir):
    vector_store_path = os.path.join(persist_dir, "default__vector_store.json")
    if not os.path.isfile(vector_store_path):
        return None

    try:
        with open(vector_store_path, "r", encoding="utf-8") as vector_store_file:
            vector_store = json.load(vector_store_file)
    except (OSError, json.JSONDecodeError):
        return None

    embedding_dict = vector_store.get("embedding_dict") or {}
    if not isinstance(embedding_dict, dict) or not embedding_dict:
        return None

    first_embedding = next(iter(embedding_dict.values()), None)
    if not isinstance(first_embedding, list):
        return None
    return len(first_embedding)


def _index_embedding_is_current(persist_dir):
    embedding_dim = _vector_store_embedding_dim(persist_dir)
    if embedding_dim is None:
        return True
    return embedding_dim == EXPECTED_EMBEDDING_DIM


def _has_index_files(persist_dir):
    return (
        os.path.isdir(persist_dir)
        and os.path.isfile(os.path.join(persist_dir, "docstore.json"))
        and os.path.isfile(os.path.join(persist_dir, "index_store.json"))
        and os.path.isfile(os.path.join(persist_dir, "default__vector_store.json"))
    )


def _glossary_docs_signature(
    glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
):
    files = []
    for root, _dirs, filenames in os.walk(glossary_base_dir):
        for filename in filenames:
            if not filename.lower().endswith((".md", ".txt")):
                continue
            files.append(os.path.join(root, filename))
    if metadata_path:
        files.append(metadata_path)

    signature = []
    for file_path in sorted(set(files)):
        if not os.path.isfile(file_path):
            signature.append((file_path, None, None))
            continue
        stat_result = os.stat(file_path)
        signature.append((file_path, stat_result.st_mtime_ns, stat_result.st_size))
    return tuple(signature)


def _get_glossary_docs_cached(
    glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
):
    signature = _glossary_docs_signature(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    cache_key = (os.path.normpath(glossary_base_dir), os.path.normpath(metadata_path or ""))
    cache_entry = _GLOSSARY_DOCS_CACHE.get(cache_key)
    if cache_entry and cache_entry["signature"] == signature:
        return cache_entry["docs"]

    docs = ingest_knowledge.build_glossary_docs(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    _GLOSSARY_DOCS_CACHE[cache_key] = {
        "signature": signature,
        "docs": docs,
    }
    return docs


def _node_text(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)
    if hasattr(node, "get_content"):
        return node.get_content().strip()
    return str(getattr(node, "text", "")).strip()


def _node_metadata(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)
    return dict(getattr(node, "metadata", {}) or {})


def _retrieval_priority(node_with_score):
    metadata = _node_metadata(node_with_score)
    doc_type = metadata.get("type")
    retrieval_tier = metadata.get("retrieval_tier")
    base_score = float(getattr(node_with_score, "score", 0.0) or 0.0)

    if retrieval_tier == "active":
        base_score += 5.0
    elif retrieval_tier == "archive":
        base_score -= 1.5

    if doc_type in {"company_profile", "sec_section_summary", "filing_financial_summary"}:
        base_score += 3.0
    elif doc_type in {
        "filing_derived_indicators",
        "filing_reported_facts",
        "statement_linked_facts",
        "financial_sector_note_summary",
    }:
        base_score += 2.0

    return base_score


def _retrieve_chunks(retriever, query_str):
    try:
        nodes = retriever.retrieve(query_str)
    except Exception as exc:
        print(f"Vector retrieval skipped: {exc}")
        return []
    nodes = sorted(nodes, key=_retrieval_priority, reverse=True)
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


def _keyword_terms(text):
    return {
        term
        for term in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(term) > 2
    }


def _fallback_stock_context(ticker, query_str, max_chunks=6):
    try:
        docs = ingest_stock.build_financial_docs(ticker)
    except Exception as exc:
        print(f"Stock doc fallback skipped for {ticker}: {exc}")
        return None

    query_terms = _keyword_terms(f"{ticker} {query_str}")
    ranked_docs = []
    for doc in docs:
        text = str(doc.text or "").strip()
        if not text:
            continue
        metadata = dict(doc.metadata or {})
        doc_terms = _keyword_terms(text[:4000])
        overlap = len(query_terms & doc_terms)
        priority = 0
        if metadata.get("type") == "company_profile":
            priority += 5
        if metadata.get("type") in {"sec_section_summary", "filing_financial_summary"}:
            priority += 4
        if metadata.get("type") in {
            "filing_derived_indicators",
            "filing_reported_facts",
            "statement_linked_facts",
            "financial_indicators",
        }:
            priority += 3
        if metadata.get("retrieval_tier") == "active":
            priority += 2
        ranked_docs.append((overlap * 10 + priority, text))

    ranked_docs.sort(key=lambda item: item[0], reverse=True)
    selected_chunks = []
    seen = set()
    for _score, text in ranked_docs:
        if text in seen:
            continue
        seen.add(text)
        selected_chunks.append(text)
        if len(selected_chunks) >= max_chunks:
            break

    return _format_context_chunks(f"{ticker} financial context", selected_chunks)


def _fallback_macro_context(query_str, max_chunks=4):
    try:
        docs = ingest_macro.build_market_environment_docs()
    except Exception as exc:
        print(f"Macro doc fallback skipped: {exc}")
        return None

    query_terms = _keyword_terms(query_str)
    ranked_docs = []
    for doc in docs:
        text = str(doc.text or "").strip()
        if not text:
            continue
        overlap = len(query_terms & _keyword_terms(text[:3000]))
        ranked_docs.append((overlap, text))

    ranked_docs.sort(key=lambda item: item[0], reverse=True)
    selected_chunks = []
    seen = set()
    for _score, text in ranked_docs:
        if text in seen:
            continue
        seen.add(text)
        selected_chunks.append(text)
        if len(selected_chunks) >= max_chunks:
            break

    return _format_context_chunks("Macro market context", selected_chunks)


def _query_needs_graph(query_str):
    normalized_query = re.sub(r"\s+", " ", (query_str or "").lower()).strip()
    if not normalized_query:
        return False

    graph_keywords = [
        "relationship",
        "related",
        "connected",
        "link",
        "driver",
        "cause",
        "why",
        "because",
        "risk factor",
        "filing",
        "note",
        "segment",
        "geographic",
        "debt maturity",
        "lease",
        "reserve",
        "impair",
        "goodwill",
        "tax rate",
        "provision",
        "definition",
        "what does",
        "what is",
        "indicator mean",
        "glossary",
        "customer",
        "supplier",
        "competitor",
    ]
    return any(keyword in normalized_query for keyword in graph_keywords)


def _query_needs_knowledge(query_str):
    normalized_query = re.sub(r"\s+", " ", (query_str or "").lower()).strip()
    if not normalized_query:
        return False

    knowledge_keywords = [
        "what is",
        "what does",
        "define",
        "definition",
        "mean",
        "glossary",
        "indicator",
        "ratio",
        "metric",
        "explain",
    ]
    return any(keyword in normalized_query for keyword in knowledge_keywords)


def _retrieve_context(retriever, query_str, label):
    chunks = _retrieve_chunks(retriever, query_str)
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

    for doc in _get_glossary_docs_cached():
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


def _ensure_macro_index_directory(macro_storage_dir):
    candidates = [macro_storage_dir]
    normalized_dir = os.path.normpath(macro_storage_dir)
    fallback_dir = os.path.join(
        os.path.dirname(normalized_dir),
        f"{os.path.basename(normalized_dir)}_refresh",
    )
    candidates.append(fallback_dir)

    for candidate in candidates:
        if _has_index_files(candidate) and _index_embedding_is_current(candidate):
            return candidate

    print("Macro index is missing or outdated; refreshing it now.")
    try:
        return ingest_macro.refresh_macro_index(
            db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
            persist_dir=macro_storage_dir,
        ) or macro_storage_dir
    except Exception as exc:
        print(f"Macro index refresh skipped: {exc}")
        return macro_storage_dir


def _ensure_stock_index_directory(ticker, stock_storage_base_dir):
    ticker = ticker.upper()
    persist_dir = os.path.join(stock_storage_base_dir, ticker)
    if _has_index_files(persist_dir) and _index_embedding_is_current(persist_dir):
        return persist_dir

    print(f"{ticker} stock index is missing or outdated; refreshing it now.")
    try:
        ingest_stock.refresh_ticker_data_and_index(
            ticker,
            storage_base_dir=stock_storage_base_dir,
        )
    except Exception as exc:
        print(f"Stock index refresh skipped for {ticker}: {exc}")
    return persist_dir


def _ensure_knowledge_index_directory(knowledge_storage_dir):
    if _has_index_files(knowledge_storage_dir) and _index_embedding_is_current(knowledge_storage_dir):
        return knowledge_storage_dir

    print("Knowledge index is missing or outdated; refreshing it now.")
    try:
        return ingest_knowledge.refresh_knowledge_index(storage_dir=knowledge_storage_dir) or knowledge_storage_dir
    except Exception as exc:
        print(f"Knowledge index refresh skipped: {exc}")
        return knowledge_storage_dir


def _ensure_graph_index_directory(
    ticker,
    graph_storage_dir,
):
    if ingest_graph.graph_index_exists(ticker=ticker, storage_dir=graph_storage_dir):
        return ingest_graph.shared_graph_persist_dir(storage_dir=graph_storage_dir)

    if ingest_graph.graph_index_exists(storage_dir=graph_storage_dir):
        print(f"{ticker} is missing from the shared graph; adding it now.")
    else:
        print("Shared graph layer is missing; building it now.")
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
    return persist_dir or ingest_graph.shared_graph_persist_dir(storage_dir=graph_storage_dir)


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
    resolved_macro_storage_dir = _ensure_macro_index_directory(macro_storage_dir)
    graph_is_needed = _query_needs_graph(query_str)
    resolved_graph_storage_dir = (
        _ensure_graph_index_directory(
            ticker,
            graph_storage_dir=graph_storage_dir,
        )
        if graph_is_needed
        else ingest_graph.shared_graph_persist_dir(storage_dir=graph_storage_dir)
    )
    try:
        stock_retriever = _load_retriever(stock_persist_dir, similarity_top_k=max(stock_top_k, 6))
    except (FileNotFoundError, ValueError):
        stock_retriever = None

    try:
        knowledge_retriever = _load_retriever(
            resolved_knowledge_storage_dir,
            similarity_top_k=knowledge_top_k,
        )
    except (FileNotFoundError, ValueError):
        knowledge_retriever = None

    try:
        macro_retriever = _load_retriever(
            resolved_macro_storage_dir,
            similarity_top_k=macro_top_k,
        )
    except (FileNotFoundError, ValueError):
        macro_retriever = None

    stock_chunks = []
    seen_stock_chunks = set()
    if stock_retriever is not None:
        stock_query = (
            f"{ticker} {query_str} "
            "company overview business description sector industry "
            "sec filing summary item 1 item 1a item 7 md&a item 2 "
            "business drivers risks reported facts statement-linked facts"
        )
        for chunk in _retrieve_chunks(stock_retriever, stock_query):
            if chunk in seen_stock_chunks:
                continue
            seen_stock_chunks.add(chunk)
            stock_chunks.append(chunk)
            if len(stock_chunks) >= 8:
                break

    stock_context = _format_context_chunks(
        f"{ticker} financial context",
        stock_chunks,
    )
    if not stock_context:
        stock_context = _fallback_stock_context(ticker, query_str)
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

    if knowledge_retriever is not None and (_query_needs_knowledge(query_str) or knowledge_chunks):
        for chunk in _retrieve_chunks(knowledge_retriever, query_str):
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
    macro_context = (
        _retrieve_context(
            macro_retriever,
            query_str,
            label="Macro market context",
        )
        if macro_retriever is not None
        else None
    )
    if not macro_context:
        macro_context = _fallback_macro_context(query_str)
    graph_context = None
    if graph_is_needed:
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
