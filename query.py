import json
import os
import re

from llama_index.core import Settings, StorageContext, PromptTemplate, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv

import ingest_knowledge
import ingest_stock

load_dotenv("config.env")

DEFAULT_STOCK_STORAGE_BASE_DIR = os.getenv("STOCK_STORAGE_BASE_DIR", "./storage/stock")
DEFAULT_KNOWLEDGE_STORAGE_DIR = os.getenv("KNOWLEDGE_STORAGE_DIR", "./storage/knowledge")
DEFAULT_MACRO_STORAGE_DIR = os.getenv("MACRO_STORAGE_DIR", "./storage/macro")


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


def _retrieve_context_from_dir(persist_dir, query_str, label, similarity_top_k=4):
    chunks = _retrieve_chunks_from_dir(
        persist_dir,
        query_str,
        similarity_top_k=similarity_top_k,
    )
    return _format_context_chunks(label, chunks)


def _matched_knowledge_chunks(query_str, max_matches=3):
    normalized_query = re.sub(r"\s+", " ", query_str.lower()).strip()
    exact_matches = []

    for doc in ingest_knowledge.build_glossary_docs():
        indicator_name = (doc.metadata.get("indicator_name") or "").strip()
        if not indicator_name:
            continue

        normalized_indicator = re.sub(r"\s+", " ", indicator_name.lower()).strip()
        if normalized_indicator and normalized_indicator in normalized_query:
            exact_matches.append((len(normalized_indicator), doc.text))

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


def _load_relationship_graph(persist_dir):
    graph_path = os.path.join(persist_dir, ingest_stock.RELATIONSHIP_GRAPH_FILENAME)
    if not os.path.isfile(graph_path):
        return None

    with open(graph_path, "r", encoding="utf-8") as graph_file:
        return json.load(graph_file)


def _query_terms(query_str):
    return {
        token
        for token in re.split(r"[^a-zA-Z0-9]+", query_str.lower())
        if len(token) >= 3
    }


def _relationship_focus(query_str):
    lowered_query = query_str.lower()
    focus = set()

    if "competitor" in lowered_query or "competition" in lowered_query:
        focus.add("competitor")
    if "supplier" in lowered_query or "suppliers" in lowered_query or "supply chain" in lowered_query:
        focus.add("supplier")
    if "customer" in lowered_query or "customers" in lowered_query or "revenue contribution" in lowered_query:
        focus.add("customer")
    return focus


def _score_graph_edge(edge, target_node, query_terms, relationship_focus):
    searchable_text = " ".join(
        [
            edge.get("relationship_type", ""),
            target_node.get("name", ""),
            target_node.get("profile", ""),
            edge.get("summary", ""),
            edge.get("quantitative_detail", ""),
            edge.get("evidence", ""),
        ]
    ).lower()

    score = 0
    for term in query_terms:
        if term in searchable_text:
            score += 1

    if relationship_focus and edge.get("relationship_type") in relationship_focus:
        score += 4

    if edge.get("quantitative_detail") and any(
        term in query_terms for term in {"revenue", "share", "margin", "contribute", "contribution"}
    ):
        score += 2

    return score


def _score_graph_node(node, query_terms, relationship_focus):
    searchable_text = " ".join(
        [
            node.get("name", ""),
            node.get("entity_type", ""),
            " ".join(node.get("roles", [])),
            node.get("profile", ""),
        ]
    ).lower()

    score = 0
    for term in query_terms:
        if term in searchable_text:
            score += 1

    if relationship_focus and set(node.get("roles", [])) & relationship_focus:
        score += 3

    return score


def _retrieve_graph_context(persist_dir, query_str, max_edges=6, max_neighbor_nodes=8):
    relationship_graph = _load_relationship_graph(persist_dir)
    if not relationship_graph:
        return None

    nodes = relationship_graph.get("nodes", [])
    edges = relationship_graph.get("edges", [])
    if not nodes:
        return None

    node_map = {node["id"]: node for node in nodes}
    query_terms = _query_terms(query_str)
    relationship_focus = _relationship_focus(query_str)

    scored_edges = []
    for edge in edges:
        target_node = node_map.get(edge.get("target_node_id"), {})
        score = _score_graph_edge(edge, target_node, query_terms, relationship_focus)
        if score > 0 or (relationship_focus and edge.get("relationship_type") in relationship_focus):
            scored_edges.append((score, edge, target_node))

    scored_edges.sort(
        key=lambda item: (
            item[0],
            item[1].get("confidence", 0.0),
            item[2].get("name", ""),
        ),
        reverse=True,
    )
    top_edges = scored_edges[:max_edges]

    related_node_ids = []
    for _, edge, _ in top_edges:
        related_node_ids.append(edge.get("target_node_id"))

    scored_nodes = []
    for node in nodes:
        if node.get("id") == f"company::{relationship_graph.get('ticker')}":
            continue
        score = _score_graph_node(node, query_terms, relationship_focus)
        if score > 0 or (relationship_focus and set(node.get("roles", [])) & relationship_focus):
            scored_nodes.append((score, node))

    scored_nodes.sort(
        key=lambda item: (item[0], item[1].get("name", "")),
        reverse=True,
    )

    related_nodes = []
    seen_node_ids = set()
    for node_id in related_node_ids:
        if not node_id or node_id in seen_node_ids:
            continue
        seen_node_ids.add(node_id)
        node = node_map.get(node_id)
        if node:
            related_nodes.append(node)

    for _, node in scored_nodes:
        if node.get("id") in seen_node_ids:
            continue
        seen_node_ids.add(node.get("id"))
        related_nodes.append(node)
        if len(related_nodes) >= max_neighbor_nodes:
            break

    if not top_edges and not related_nodes:
        return None

    graph_lines = [
        f"Company: {relationship_graph.get('company_name', relationship_graph.get('ticker'))}",
        f"Ticker: {relationship_graph.get('ticker')}",
    ]

    if top_edges:
        graph_lines.append("")
        graph_lines.append("Top relationship matches:")
        for _, edge, target_node in top_edges:
            graph_lines.append(
                f"- {edge.get('relationship_type', 'unknown')} -> {target_node.get('name', 'Unknown')}: "
                f"{edge.get('summary') or 'No short summary extracted.'}"
            )
            if edge.get("quantitative_detail"):
                graph_lines.append(f"  Quantitative detail: {edge['quantitative_detail']}")
            if edge.get("evidence"):
                graph_lines.append(f"  Evidence: {edge['evidence']}")

    if related_nodes:
        graph_lines.append("")
        graph_lines.append("Related entities:")
        for node in related_nodes[:max_neighbor_nodes]:
            graph_lines.append(
                f"- {node.get('name', 'Unknown')} | Roles: {', '.join(node.get('roles', [])) or 'Unknown'} | "
                f"Profile: {node.get('profile') or 'No short profile extracted.'}"
            )

    return "Relationship graph context:\n" + "\n".join(graph_lines)


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


def get_analysis_context(
    ticker,
    query_str,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    knowledge_storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    stock_top_k=4,
    knowledge_top_k=3,
    macro_top_k=4):
    ingest_stock.env()

    ticker = ticker.upper()
    stock_persist_dir = _ensure_stock_index_directory(
        ticker,
        stock_storage_base_dir=stock_storage_base_dir,
    )
    resolved_knowledge_storage_dir = _ensure_knowledge_index_directory(knowledge_storage_dir)
    resolved_macro_storage_dir = _resolve_macro_storage_dir(macro_storage_dir)

    stock_chunks = []
    seen_stock_chunks = set()
    stock_queries = [
        f"{ticker} company overview business description sector industry",
        f"{ticker} financial indicators revenue margins eps cash flow balance sheet leverage quarterly annual",
        f"{ticker} sec 10-k 10-q item 1 item 1a item 7 item 8 md&a business risk factors financial statements",
        query_str,
    ]
    for stock_query in stock_queries:
        for chunk in _retrieve_chunks_from_dir(
            stock_persist_dir,
            stock_query,
            similarity_top_k=stock_top_k,
        ):
            if chunk in seen_stock_chunks:
                continue
            seen_stock_chunks.add(chunk)
            stock_chunks.append(chunk)
            if len(stock_chunks) >= 8:
                break
        if len(stock_chunks) >= 8:
            break

    stock_context = _format_context_chunks(
        f"{ticker} financial context",
        stock_chunks,
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

    return {
        "stock_context": stock_context,
        "knowledge_context": knowledge_context,
        "macro_context": macro_context,
        "stock_persist_dir": stock_persist_dir,
        "knowledge_persist_dir": resolved_knowledge_storage_dir,
        "macro_persist_dir": resolved_macro_storage_dir,
    }


def analyze_company(
    ticker,
    custom_query=None,
    stock_storage_base_dir=DEFAULT_STOCK_STORAGE_BASE_DIR,
    knowledge_storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
    macro_storage_dir=DEFAULT_MACRO_STORAGE_DIR,
    stock_top_k=4,
    knowledge_top_k=3,
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
        knowledge_storage_dir=knowledge_storage_dir,
        macro_storage_dir=macro_storage_dir,
        stock_top_k=stock_top_k,
        knowledge_top_k=knowledge_top_k,
        macro_top_k=macro_top_k,
    )

    if not context["stock_context"] and not context["knowledge_context"] and not context["macro_context"]:
        raise FileNotFoundError(
            "No retrievable stock, knowledge, or macro index was found. Refresh the stock index "
            f"at {context['stock_persist_dir']}, the knowledge index at {context['knowledge_persist_dir']}, and the macro index at "
            f"{context['macro_persist_dir']} first."
        )

    stock_context = context["stock_context"] or "Stock-specific context was not available."
    knowledge_context = context["knowledge_context"] or "Financial glossary context was not available."
    macro_context = context["macro_context"] or "Macro market context was not available."

    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        question=query_str,
        stock_context=stock_context,
        knowledge_context=knowledge_context,
        macro_context=macro_context,
    )

    response = Settings.llm.complete(prompt)
    return str(response)
