import json
import os
import re
import shutil
import sqlite3
import stat
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

import ingest_knowledge
import ingest_macro
import ingest_stock


load_dotenv("config.env")

DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")
GRAPH_STATE_FILENAME = "graph_state.json"

MACRO_GRAPH_CONCEPT_NAMES = {
    "fed_funds_rate": "Fed interest rate",
    "real_gdp": "GDP",
    "cpi_all_items": "CPI",
    "cpi_inflation_yoy": "Inflation rate",
    "unemployment_rate": "Unemployment rate",
    "adp_private_payrolls": "ADP",
    "nonfarm_payrolls": "BLS",
    "pmi": "PMI",
}


def _slug(value):
    normalized_value = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return normalized_value or "unknown"


def _trim_text(text, max_chars=4000):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _graph_state_path(storage_dir):
    return os.path.join(storage_dir, GRAPH_STATE_FILENAME)


def _empty_graph_state():
    return {
        "nodes": {},
        "edges": [],
        "updated_at": None,
    }


def _load_graph_state(storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    state_path = _graph_state_path(storage_dir)
    if not os.path.isfile(state_path):
        return _empty_graph_state()

    with open(state_path, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    state.setdefault("nodes", {})
    state.setdefault("edges", [])
    return state


def _save_graph_state(state, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    os.makedirs(storage_dir, exist_ok=True)
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(_graph_state_path(storage_dir), "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)


def _edge_key(edge):
    return (
        edge["source"],
        edge["target"],
        edge["type"],
        edge.get("scope", ""),
    )


def _remove_scope(state, scope):
    state["nodes"] = {
        node_id: node
        for node_id, node in state["nodes"].items()
        if node.get("scope") != scope
    }
    state["edges"] = [
        edge
        for edge in state["edges"]
        if edge.get("scope") != scope
        and edge["source"] in state["nodes"]
        and edge["target"] in state["nodes"]
    ]


def _merge_scope(state, scope, nodes, edges):
    _remove_scope(state, scope)
    for node in nodes:
        state["nodes"][node["id"]] = node

    existing_edge_keys = {_edge_key(edge) for edge in state["edges"]}
    for edge in edges:
        if edge["source"] not in state["nodes"] or edge["target"] not in state["nodes"]:
            continue
        edge_key = _edge_key(edge)
        if edge_key in existing_edge_keys:
            continue
        state["edges"].append(edge)
        existing_edge_keys.add(edge_key)

    state["edges"] = [
        edge
        for edge in state["edges"]
        if edge["source"] in state["nodes"] and edge["target"] in state["nodes"]
    ]
    return state


def _adjacency_map(state):
    adjacency = {}
    for edge in state.get("edges", []):
        adjacency.setdefault(edge["source"], []).append(edge)
        adjacency.setdefault(edge["target"], []).append(edge)
    return adjacency


def _node_neighbors(state, node_id):
    neighbors = []
    for edge in _adjacency_map(state).get(node_id, []):
        other_id = edge["target"] if edge["source"] == node_id else edge["source"]
        other_node = state["nodes"].get(other_id)
        if not other_node:
            continue
        neighbors.append((edge, other_node))
    return neighbors


def _node_to_document(state, node):
    node_id = node["id"]
    graph_connections = []
    for edge, other_node in _node_neighbors(state, node_id)[:10]:
        graph_connections.append(f"- {edge['type']}: {other_node['label']}")

    text = node["text"]
    if graph_connections:
        text = text + "\n\nGraph Connections:\n" + "\n".join(graph_connections)

    metadata = dict(node.get("metadata", {}))
    metadata.update(
        {
            "graph_node_id": node_id,
            "graph_node_type": node["type"],
            "graph_scope": node["scope"],
            "graph_label": node["label"],
            "graph_domain": node.get("domain"),
        }
    )
    return Document(text=text, metadata=metadata)


def _graph_documents_from_state(state):
    documents = []
    for node_id in sorted(state.get("nodes", {})):
        documents.append(_node_to_document(state, state["nodes"][node_id]))
    return documents


def _clear_windows_readonly(path):
    if os.name != "nt" or not os.path.lexists(path):
        return
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _handle_remove_readonly(func, path, exc_info):
    _clear_windows_readonly(path)
    func(path)


def _is_windows_reparse_point(path):
    if os.name != "nt" or not os.path.lexists(path):
        return False
    try:
        return bool(os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    except (AttributeError, OSError):
        return False


def _reset_persist_dir(persist_dir):
    if os.path.lexists(persist_dir):
        _clear_windows_readonly(persist_dir)
        if _is_windows_reparse_point(persist_dir):
            try:
                os.rmdir(persist_dir)
            except OSError:
                shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
        else:
            shutil.rmtree(persist_dir, onerror=_handle_remove_readonly)
    os.makedirs(persist_dir, exist_ok=True)


def _persist_graph_index(state, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    ingest_stock.env()
    documents = _graph_documents_from_state(state)
    if not documents:
        return None

    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(documents)
    _reset_persist_dir(storage_dir)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=storage_dir)
    _save_graph_state(state, storage_dir=storage_dir)
    return storage_dir


def _indicator_node_id(domain, indicator_name):
    return f"indicator::{domain}::{_slug(indicator_name)}"


def _make_node(node_id, label, node_type, scope, text, metadata=None, domain=None):
    return {
        "id": node_id,
        "label": label,
        "type": node_type,
        "scope": scope,
        "domain": domain,
        "text": text,
        "metadata": metadata or {},
    }


def _make_edge(source, target, edge_type, scope, metadata=None):
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "scope": scope,
        "metadata": metadata or {},
    }


def _build_knowledge_graph_elements(docs):
    scope = "knowledge"
    nodes = []
    edges = []

    for doc in docs:
        metadata = doc.metadata or {}
        indicator_name = (
            metadata.get("indicator_canonical_name")
            or metadata.get("indicator_name")
            or "Unknown Indicator"
        )
        glossary_domain = metadata.get("glossary_domain") or "general"
        group = metadata.get("group") or ("macro" if glossary_domain == "eco" else "unmapped")
        subgroup = metadata.get("subgroup") or ("macro" if glossary_domain == "eco" else "unmapped")
        indicator_aliases = metadata.get("indicator_aliases") or []

        group_node_id = f"group::{glossary_domain}::{_slug(group)}"
        subgroup_node_id = f"subgroup::{glossary_domain}::{_slug(group)}::{_slug(subgroup)}"
        indicator_node_id = _indicator_node_id(glossary_domain, indicator_name)
        glossary_node_id = f"glossary_doc::{glossary_domain}::{_slug(indicator_name)}"

        nodes.append(
            _make_node(
                group_node_id,
                group,
                "indicator_group",
                scope,
                f"Indicator Group: {group}\nDomain: {glossary_domain}",
                metadata={
                    "group": group,
                    "glossary_domain": glossary_domain,
                },
                domain=glossary_domain,
            )
        )
        nodes.append(
            _make_node(
                subgroup_node_id,
                subgroup,
                "indicator_subgroup",
                scope,
                f"Indicator Subgroup: {subgroup}\nGroup: {group}\nDomain: {glossary_domain}",
                metadata={
                    "group": group,
                    "subgroup": subgroup,
                    "glossary_domain": glossary_domain,
                },
                domain=glossary_domain,
            )
        )
        nodes.append(
            _make_node(
                indicator_node_id,
                indicator_name,
                "indicator_concept",
                scope,
                (
                    f"Indicator Concept: {indicator_name}\n"
                    f"Domain: {glossary_domain}\n"
                    f"Group: {group}\n"
                    f"Subgroup: {subgroup}\n"
                    f"Aliases: {', '.join(indicator_aliases) if indicator_aliases else 'None'}\n\n"
                    f"{_trim_text(doc.text)}"
                ),
                metadata={
                    "indicator_name": indicator_name,
                    "indicator_aliases": indicator_aliases,
                    "group": group,
                    "subgroup": subgroup,
                    "glossary_domain": glossary_domain,
                    "source": metadata.get("source"),
                },
                domain=glossary_domain,
            )
        )
        nodes.append(
            _make_node(
                glossary_node_id,
                f"{indicator_name} glossary",
                "glossary_document",
                scope,
                doc.text,
                metadata=dict(metadata),
                domain=glossary_domain,
            )
        )

        edges.extend(
            [
                _make_edge(group_node_id, subgroup_node_id, "contains_subgroup", scope),
                _make_edge(subgroup_node_id, indicator_node_id, "contains_indicator", scope),
                _make_edge(indicator_node_id, glossary_node_id, "defined_by", scope),
            ]
        )

    return nodes, edges


def _latest_macro_rows(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM macro_indicators ORDER BY observation_date DESC",
        conn,
    )
    conn.close()

    if df.empty:
        return []

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"])
    latest_df = (
        df.sort_values("observation_date", ascending=False)
        .groupby("indicator_key", as_index=False)
        .first()
    )
    return latest_df.to_dict(orient="records")


def _build_macro_graph_elements(docs, db_path):
    scope = "macro"
    nodes = []
    edges = []
    macro_root_id = "macro_environment::global"

    nodes.append(
        _make_node(
            macro_root_id,
            "Macro Environment",
            "macro_environment",
            scope,
            "Global macro environment node connected to glossary concepts, macro documents, and the latest macro observations.",
            metadata={"type": "macro_environment"},
            domain="eco",
        )
    )

    for doc in docs:
        metadata = doc.metadata or {}
        node_id = f"macro_doc::{_slug(metadata.get('type') or 'doc')}::{_slug(metadata.get('category') or 'general')}"
        label = metadata.get("category") or metadata.get("type") or "Macro Document"
        nodes.append(
            _make_node(
                node_id,
                label,
                "macro_document",
                scope,
                doc.text,
                metadata=dict(metadata),
                domain="eco",
            )
        )
        edges.append(_make_edge(macro_root_id, node_id, "described_by", scope))

    for row in _latest_macro_rows(db_path):
        indicator_key = row["indicator_key"]
        concept_name = MACRO_GRAPH_CONCEPT_NAMES.get(indicator_key, row["indicator_name"])
        observation_id = (
            f"macro_observation::{indicator_key}::"
            f"{row['observation_date'].strftime('%Y-%m-%d')}"
        )
        concept_id = _indicator_node_id("eco", concept_name)
        value = row.get("value")
        units = row.get("units") or ""

        nodes.append(
            _make_node(
                observation_id,
                f"{row['indicator_name']} latest observation",
                "macro_observation",
                scope,
                (
                    f"Macro Observation\n"
                    f"Indicator: {row['indicator_name']}\n"
                    f"Glossary Indicator: {concept_name}\n"
                    f"Observation Date: {row['observation_date'].strftime('%Y-%m-%d')}\n"
                    f"Value: {value}\n"
                    f"Units: {units}\n"
                    f"Release: {row.get('release_name') or 'Unknown'}\n"
                    f"Source: {row.get('source') or 'Unknown'}"
                ),
                metadata={
                    "indicator_key": indicator_key,
                    "indicator_name": row["indicator_name"],
                    "glossary_indicator_name": concept_name,
                    "observation_date": row["observation_date"].strftime("%Y-%m-%d"),
                    "value": value,
                    "units": units,
                    "release_name": row.get("release_name"),
                    "source": row.get("source"),
                },
                domain="eco",
            )
        )
        edges.append(_make_edge(macro_root_id, observation_id, "has_observation", scope))
        edges.append(_make_edge(observation_id, concept_id, "measures", scope))

    return nodes, edges


def _latest_stock_rows(ticker, db_path):
    conn = sqlite3.connect(db_path)
    latest_rows = []
    for frequency in ("Quarterly", "Annual"):
        df = pd.read_sql_query(
            """
            SELECT *
            FROM financial_indicators
            WHERE Ticker = ? AND Frequency = ?
            ORDER BY `Period End Date` DESC
            LIMIT 1
            """,
            conn,
            params=(ticker.upper(), frequency),
        )
        if not df.empty:
            latest_rows.append(df.iloc[0].to_dict())
    conn.close()
    return latest_rows


def _stock_doc_node_id(ticker, doc, index_position):
    metadata = doc.metadata or {}
    doc_type = metadata.get("type") or "stock_document"
    if doc_type == "company_profile":
        return f"stock_doc::{ticker}::company_profile"
    if doc_type == "financial_indicators":
        return f"stock_doc::{ticker}::financial_indicators::{_slug(metadata.get('frequency') or 'general')}"
    if doc_type == "sec_filing_section":
        return (
            f"stock_doc::{ticker}::{_slug(metadata.get('form_type') or 'filing')}::"
            f"{_slug(metadata.get('section_key') or 'section')}::"
            f"{_slug(metadata.get('filing_date') or index_position)}"
        )
    return f"stock_doc::{ticker}::{_slug(doc_type)}::{index_position}"


def _build_stock_graph_elements(ticker, docs, db_path):
    scope = f"stock:{ticker.upper()}"
    nodes = []
    edges = []
    ticker = ticker.upper()

    company_name = ticker
    sector = "Unknown"
    industry = "Unknown"
    for doc in docs:
        metadata = doc.metadata or {}
        if metadata.get("type") == "company_profile":
            company_name = metadata.get("company_name") or company_name
            sector = metadata.get("sector") or sector
            industry = metadata.get("industry") or industry
            break

    company_node_id = f"company::{ticker}"
    nodes.append(
        _make_node(
            company_node_id,
            company_name,
            "company",
            scope,
            (
                f"Company: {company_name}\n"
                f"Ticker: {ticker}\n"
                f"Sector: {sector}\n"
                f"Industry: {industry}"
            ),
            metadata={
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
            },
            domain="company",
        )
    )

    for index_position, doc in enumerate(docs):
        metadata = dict(doc.metadata or {})
        metadata.setdefault("ticker", ticker)
        node_id = _stock_doc_node_id(ticker, doc, index_position)
        label = (
            metadata.get("section_title")
            or metadata.get("type")
            or f"{ticker} document"
        )
        nodes.append(
            _make_node(
                node_id,
                label,
                "stock_document",
                scope,
                doc.text,
                metadata=metadata,
                domain="company",
            )
        )
        edges.append(_make_edge(company_node_id, node_id, "has_document", scope))

    for row in _latest_stock_rows(ticker, db_path):
        frequency = row.get("Frequency") or "Unknown"
        period_end_date = row.get("Period End Date") or "Unknown"
        for indicator_name in ingest_stock.CORE_GLOSSARY_INDICATORS:
            value = row.get(indicator_name)
            if pd.isna(value):
                continue

            observation_id = (
                f"stock_observation::{ticker}::{_slug(frequency)}::"
                f"{_slug(period_end_date)}::{_slug(indicator_name)}"
            )
            concept_id = _indicator_node_id("company", indicator_name)
            nodes.append(
                _make_node(
                    observation_id,
                    f"{indicator_name} latest {frequency.lower()} observation",
                    "stock_observation",
                    scope,
                    (
                        f"Financial Observation\n"
                        f"Company: {company_name}\n"
                        f"Ticker: {ticker}\n"
                        f"Indicator: {indicator_name}\n"
                        f"Frequency: {frequency}\n"
                        f"Period End Date: {period_end_date}\n"
                        f"Value: {value}"
                    ),
                    metadata={
                        "ticker": ticker,
                        "company_name": company_name,
                        "indicator_name": indicator_name,
                        "frequency": frequency,
                        "period_end_date": period_end_date,
                        "value": value,
                        "sector": sector,
                        "industry": industry,
                    },
                    domain="company",
                )
            )
            edges.append(_make_edge(company_node_id, observation_id, "has_observation", scope))
            edges.append(_make_edge(observation_id, concept_id, "measures", scope))

    return nodes, edges


def refresh_knowledge_graph(docs=None, storage_dir=DEFAULT_GRAPH_STORAGE_DIR, rebuild_index=True):
    docs = docs or ingest_knowledge.build_glossary_docs()
    state = _load_graph_state(storage_dir=storage_dir)
    nodes, edges = _build_knowledge_graph_elements(docs)
    state = _merge_scope(state, "knowledge", nodes, edges)
    if rebuild_index:
        _persist_graph_index(state, storage_dir=storage_dir)
    else:
        _save_graph_state(state, storage_dir=storage_dir)
    return storage_dir


def refresh_macro_graph(
    docs=None,
    db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    rebuild_index=True,
):
    docs = docs or ingest_macro.build_market_environment_docs(db_path=db_path)
    state = _load_graph_state(storage_dir=storage_dir)
    nodes, edges = _build_macro_graph_elements(docs, db_path=db_path)
    state = _merge_scope(state, "macro", nodes, edges)
    if rebuild_index:
        _persist_graph_index(state, storage_dir=storage_dir)
    else:
        _save_graph_state(state, storage_dir=storage_dir)
    return storage_dir


def refresh_stock_graph(
    ticker,
    docs,
    db_path=ingest_stock.DEFAULT_STOCK_DB_PATH,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    rebuild_index=True,
):
    state = _load_graph_state(storage_dir=storage_dir)
    nodes, edges = _build_stock_graph_elements(ticker, docs, db_path=db_path)
    state = _merge_scope(state, f"stock:{ticker.upper()}", nodes, edges)
    if rebuild_index:
        _persist_graph_index(state, storage_dir=storage_dir)
    else:
        _save_graph_state(state, storage_dir=storage_dir)
    return storage_dir


def refresh_full_graph_for_ticker(
    ticker,
    stock_docs=None,
    stock_db_path=ingest_stock.DEFAULT_STOCK_DB_PATH,
    macro_db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
    filings_base_dir=ingest_stock.DEFAULT_STOCK_FILINGS_BASE_DIR,
    glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    ingest_macro.refresh_market_environment_if_stale(db_path=macro_db_path)
    knowledge_docs = ingest_knowledge.build_glossary_docs(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    macro_docs = ingest_macro.build_market_environment_docs(db_path=macro_db_path)
    if stock_docs is None:
        stock_docs = ingest_stock.build_financial_docs(
            ticker,
            db_path=stock_db_path,
            filings_base_dir=filings_base_dir,
        )

    state = _load_graph_state(storage_dir=storage_dir)
    knowledge_nodes, knowledge_edges = _build_knowledge_graph_elements(knowledge_docs)
    macro_nodes, macro_edges = _build_macro_graph_elements(macro_docs, db_path=macro_db_path)
    stock_nodes, stock_edges = _build_stock_graph_elements(ticker, stock_docs, db_path=stock_db_path)

    state = _merge_scope(state, "knowledge", knowledge_nodes, knowledge_edges)
    state = _merge_scope(state, "macro", macro_nodes, macro_edges)
    state = _merge_scope(state, f"stock:{ticker.upper()}", stock_nodes, stock_edges)
    _persist_graph_index(state, storage_dir=storage_dir)
    return storage_dir


def graph_index_exists(storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    return (
        os.path.isdir(storage_dir)
        and os.path.isfile(os.path.join(storage_dir, "docstore.json"))
        and os.path.isfile(_graph_state_path(storage_dir))
    )


def _graph_retriever(storage_dir=DEFAULT_GRAPH_STORAGE_DIR, similarity_top_k=4):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    return index.as_retriever(similarity_top_k=similarity_top_k)


def _graph_hit_ids(retrieved_nodes, ticker=None):
    hit_ids = []
    for retrieved_node in retrieved_nodes:
        node = getattr(retrieved_node, "node", retrieved_node)
        metadata = getattr(node, "metadata", {}) or {}
        graph_node_id = metadata.get("graph_node_id")
        node_ticker = (metadata.get("ticker") or "").upper()
        if ticker and node_ticker and node_ticker != ticker.upper():
            continue
        if graph_node_id and graph_node_id not in hit_ids:
            hit_ids.append(graph_node_id)
    return hit_ids


def _format_graph_section(state, node_id, ticker=None):
    node = state["nodes"].get(node_id)
    if not node:
        return None

    lines = [
        f"Node: {node['label']}",
        f"Type: {node['type']}",
        _trim_text(node["text"], max_chars=800),
    ]

    connections = []
    for edge, other_node in _node_neighbors(state, node_id)[:8]:
        other_ticker = (other_node.get("metadata", {}).get("ticker") or "").upper()
        if ticker and other_ticker and other_ticker != ticker.upper():
            continue
        connections.append(f"- {edge['type']}: {other_node['label']}")

    if connections:
        lines.append("Related Graph Connections:")
        lines.extend(connections)

    return "\n".join(lines)


def _graph_node_priority(node_type):
    priorities = {
        "stock_observation": 0,
        "macro_observation": 0,
        "indicator_concept": 1,
        "glossary_document": 2,
        "company": 3,
        "macro_environment": 3,
        "stock_document": 4,
        "macro_document": 4,
        "indicator_group": 5,
        "indicator_subgroup": 5,
    }
    return priorities.get(node_type, 9)


def retrieve_graph_context(
    query_str,
    ticker=None,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    similarity_top_k=4,
):
    if not graph_index_exists(storage_dir=storage_dir):
        return None

    retriever = _graph_retriever(storage_dir=storage_dir, similarity_top_k=similarity_top_k)
    retrieved_nodes = retriever.retrieve(query_str)
    hit_ids = _graph_hit_ids(retrieved_nodes, ticker=ticker)
    if not hit_ids:
        return None

    state = _load_graph_state(storage_dir=storage_dir)
    hit_ids = sorted(
        hit_ids,
        key=lambda node_id: _graph_node_priority(
            state["nodes"].get(node_id, {}).get("type", "")
        ),
    )
    sections = []
    for node_id in hit_ids[:3]:
        section = _format_graph_section(state, node_id, ticker=ticker)
        if section:
            sections.append(section)

    if not sections:
        return None

    return "Graph layer context:\n\n" + "\n\n".join(sections)
