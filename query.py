import json
import os
import re

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

        === RELATIONSHIP GRAPH CONTEXT ===
        {graph_context}

        === MACRO DATABASE CONTEXT ===
        {macro_context}

        The stock context may contain:
        - company overview / business description
        - sector and industry
        - annual and quarterly financial indicators

        The relationship graph context may contain:
        - competitors
        - suppliers
        - customers
        - relation summaries
        - quantitative details like revenue contribution if explicitly extracted

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
        - For relationship questions, use the relationship graph context first and then connect it back to the financial and macro context.

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

        4. Relationship Graph Takeaways
        - Relevant competitors, suppliers, or customers from the retrieved graph context
        - Any quantitative contribution, concentration, or relation detail explicitly present
        - If the graph does not contain enough evidence for the question, say that directly

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

    stock_chunks = []
    seen_stock_chunks = set()
    stock_queries = [
        query_str,
        f"{ticker} company overview business description sector industry",
        f"{ticker} sec 10-k item 1 business item 1a risk factors competition customers suppliers business model",
        f"{ticker} financial indicators revenue margins eps cash flow balance sheet leverage quarterly annual",
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

    stock_context = _format_context_chunks(
        f"{ticker} financial context",
        stock_chunks,
    )
    graph_context = _retrieve_graph_context(
        stock_persist_dir,
        query_str,
    )
    macro_context = _retrieve_context_from_dir(
        resolved_macro_storage_dir,
        query_str,
        label="Macro market context",
        similarity_top_k=macro_top_k,
    )

    return {
        "stock_context": stock_context,
        "graph_context": graph_context,
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

    if not context["stock_context"] and not context["graph_context"] and not context["macro_context"]:
        raise FileNotFoundError(
            "No retrievable stock or macro index was found. Refresh the stock index "
            f"at {context['stock_persist_dir']} and the macro index at "
            f"{context['macro_persist_dir']} first."
        )

    stock_context = context["stock_context"] or "Stock-specific context was not available."
    graph_context = context["graph_context"] or "Relationship graph context was not available."
    macro_context = context["macro_context"] or "Macro market context was not available."

    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        question=query_str,
        stock_context=stock_context,
        graph_context=graph_context,
        macro_context=macro_context,
    )

    response = Settings.llm.complete(prompt)
    return str(response)
