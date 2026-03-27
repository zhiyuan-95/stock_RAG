import json
import os
import shutil
import sqlite3
import stat

import pandas as pd
from dotenv import load_dotenv
from fsspec.implementations.local import LocalFileSystem
from llama_index.core import Document, StorageContext, load_index_from_storage
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    DynamicLLMPathExtractor,
    ImplicitPathExtractor,
    PropertyGraphIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

import ingest_knowledge
import ingest_macro
import ingest_stock


load_dotenv("config.env")

DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")
_GRAPH_LLM = None

GRAPH_ENTITY_TYPES = [
    "company",
    "indicator",
    "macro_indicator",
    "financial_indicator",
    "glossary_concept",
    "group",
    "subgroup",
    "filing",
    "filing_section",
    "observation",
    "period",
    "source",
    "risk",
    "business_driver",
]

GRAPH_RELATION_TYPES = [
    "belongs_to",
    "defines",
    "describes",
    "explains",
    "measures",
    "reported_in",
    "has_observation",
    "affects",
    "references",
    "related_to",
    "released_by",
    "sourced_from",
    "covers",
]

GRAPH_ENTITY_PROPS = [
    "ticker",
    "indicator_name",
    "group",
    "subgroup",
    "glossary_domain",
    "form_type",
    "section_key",
    "section_title",
    "filing_date",
    "frequency",
    "period_end_date",
    "observation_date",
    "value",
    "units",
    "source",
]

GRAPH_RELATION_PROPS = [
    "source_type",
    "evidence",
]


def _remove_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _reset_persist_dir(persist_dir):
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir, onerror=_remove_readonly)
    os.makedirs(persist_dir, exist_ok=True)


def _graph_persist_dir(ticker, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    return os.path.join(storage_dir, ticker.upper())


class _UTF8LocalFileSystem(LocalFileSystem):
    def open(self, path, mode="rb", **kwargs):
        if "b" not in mode:
            kwargs.setdefault("encoding", "utf-8")
        return super().open(path, mode=mode, **kwargs)


def graph_index_exists(ticker, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    persist_dir = _graph_persist_dir(ticker, storage_dir=storage_dir)
    return (
        os.path.isdir(persist_dir)
        and os.path.isfile(os.path.join(persist_dir, "property_graph_store.json"))
        and os.path.isfile(os.path.join(persist_dir, "index_store.json"))
    )


def _normalize_indicator_key(value):
    normalized = str(value or "").lower()
    normalized = "".join(ch if ch.isalnum() else " " for ch in normalized)
    return " ".join(normalized.split())


def _stringify_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    cleaned = str(value).strip()
    return cleaned or None


def _metadata_preface(metadata, preferred_keys=None):
    if not metadata:
        return ""

    ordered_keys = []
    if preferred_keys:
        ordered_keys.extend(key for key in preferred_keys if key in metadata)
    ordered_keys.extend(key for key in sorted(metadata) if key not in ordered_keys)

    lines = []
    for key in ordered_keys:
        value = _stringify_metadata_value(metadata.get(key))
        if not value:
            continue
        label = key.replace("_", " ").title()
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _prepare_graph_document(document, source_type):
    metadata = dict(document.metadata or {})
    metadata.setdefault("graph_source_type", source_type)

    preface = _metadata_preface(
        metadata,
        preferred_keys=[
            "graph_source_type",
            "type",
            "ticker",
            "company_name",
            "indicator_name",
            "group",
            "subgroup",
            "glossary_domain",
            "form_type",
            "section_key",
            "section_title",
            "filing_date",
            "frequency",
            "period_end_date",
            "observation_date",
            "value",
            "units",
            "source",
        ],
    )
    body = (document.text or "").strip()
    if source_type == "stock_context" and metadata.get("type") == "sec_filing_section":
        body = body[:4000].strip()
    elif source_type == "stock_context":
        body = body[:6000].strip()
    else:
        body = body[:5000].strip()
    text_parts = []
    if preface:
        text_parts.append(preface)
    if body:
        text_parts.append(body)
    return Document(
        text="\n\n".join(text_parts).strip(),
        metadata=metadata,
    )


def _glossary_indicator_lookup(glossary_docs):
    lookup = {}
    for doc in glossary_docs:
        metadata = doc.metadata or {}
        candidate_names = []
        for key in ("indicator_name", "indicator_canonical_name"):
            value = metadata.get(key)
            if value:
                candidate_names.append(value)
        candidate_names.extend(metadata.get("indicator_aliases", []))

        for candidate in candidate_names:
            normalized_candidate = _normalize_indicator_key(candidate)
            if not normalized_candidate or normalized_candidate in lookup:
                continue
            lookup[normalized_candidate] = {
                "indicator_name": metadata.get("indicator_name") or metadata.get("indicator_canonical_name"),
                "group": metadata.get("group"),
                "subgroup": metadata.get("subgroup"),
                "glossary_domain": metadata.get("glossary_domain"),
            }
    return lookup


def _lookup_indicator_metadata(indicator_name, glossary_lookup, fallback_group=None, fallback_subgroup=None, fallback_domain=None):
    metadata = glossary_lookup.get(_normalize_indicator_key(indicator_name), {}).copy()
    if fallback_group and not metadata.get("group"):
        metadata["group"] = fallback_group
    if fallback_subgroup and not metadata.get("subgroup"):
        metadata["subgroup"] = fallback_subgroup
    if fallback_domain and not metadata.get("glossary_domain"):
        metadata["glossary_domain"] = fallback_domain
    return metadata


def _company_profile_metadata(stock_docs, ticker):
    default_profile = {
        "ticker": ticker,
        "company_name": ticker,
        "sector": None,
        "industry": None,
    }

    for doc in stock_docs:
        metadata = doc.metadata or {}
        if metadata.get("type") != "company_profile":
            continue
        return {
            "ticker": ticker,
            "company_name": metadata.get("company_name") or ticker,
            "sector": metadata.get("sector"),
            "industry": metadata.get("industry"),
        }

    return default_profile


def _select_stock_docs_for_graph(stock_docs):
    selected_docs = []
    sec_docs_by_form = {"10-K": {}, "10-Q": {}}
    excluded_section_keys = {
        "item_8_financial_statements",
        "item_1_financial_statements",
    }

    for doc in stock_docs:
        metadata = doc.metadata or {}
        doc_type = metadata.get("type")
        if doc_type == "financial_indicators":
            continue
        if doc_type != "sec_filing_section":
            selected_docs.append(doc)
            continue

        form_type = metadata.get("form_type")
        filing_date = metadata.get("filing_date")
        section_key = metadata.get("section_key")
        if section_key in excluded_section_keys:
            continue
        if form_type not in sec_docs_by_form or not filing_date:
            selected_docs.append(doc)
            continue
        sec_docs_by_form[form_type].setdefault(filing_date, []).append(doc)

    for form_type, max_filings in (("10-K", 2), ("10-Q", 4)):
        filing_map = sec_docs_by_form[form_type]
        for filing_date in sorted(filing_map.keys(), reverse=True)[:max_filings]:
            selected_docs.extend(filing_map[filing_date])

    return selected_docs


def _latest_stock_rows(ticker, db_path):
    if not os.path.isfile(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    latest_rows = {}
    try:
        for frequency in ("Annual", "Quarterly"):
            query = """
                SELECT *
                FROM financial_indicators
                WHERE Ticker = ? AND Frequency = ?
                ORDER BY date([Period End Date]) DESC
                LIMIT 1
            """
            df = pd.read_sql_query(query, conn, params=(ticker, frequency))
            if not df.empty:
                latest_rows[frequency] = df.iloc[0]
    finally:
        conn.close()
    return latest_rows


def _latest_macro_rows(db_path):
    if not os.path.isfile(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM macro_indicators
            ORDER BY date(observation_date) DESC
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"])
    if df.empty:
        return df

    latest_rows = (
        df.sort_values("observation_date", ascending=False)
        .groupby("indicator_key", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return latest_rows


def _stock_observation_documents(ticker, db_path, glossary_lookup, stock_docs):
    latest_rows = _latest_stock_rows(ticker, db_path)
    if not latest_rows:
        return []

    company_profile = _company_profile_metadata(stock_docs, ticker)
    documents = []
    excluded_columns = set(ingest_stock.IDENTIFIER_COLUMNS) | {
        "_Period Start Date",
        "_Duration Days",
        "_Filed Date",
    }
    glossary_indicator_names = {
        metadata.get("indicator_name")
        for metadata in glossary_lookup.values()
        if metadata.get("glossary_domain") == "company" and metadata.get("indicator_name")
    }

    for frequency, row in latest_rows.items():
        period_end_date = row.get("Period End Date")
        observation_lines = []
        for column_name, value in row.items():
            if column_name in excluded_columns or column_name.startswith("_"):
                continue
            if pd.isna(value):
                continue
            if column_name not in glossary_indicator_names:
                continue

            indicator_metadata = _lookup_indicator_metadata(
                column_name,
                glossary_lookup,
                fallback_group="company",
                fallback_subgroup="reported_facts",
                fallback_domain="company",
            )
            observation_lines.append(
                (
                    f"Indicator: {column_name} | Value: {value} | Group: {indicator_metadata.get('group') or 'company'} "
                    f"| Subgroup: {indicator_metadata.get('subgroup') or 'reported_facts'}"
                )
            )

        if not observation_lines:
            continue

        lines = [
            "Observation Type: Latest stock financial indicator snapshot",
            f"Company: {company_profile['company_name']}",
            f"Ticker: {ticker}",
            f"Sector: {company_profile['sector'] or 'Unknown'}",
            f"Industry: {company_profile['industry'] or 'Unknown'}",
            f"Frequency: {frequency}",
            f"Period End Date: {period_end_date}",
            "Observation Source: SEC Company Facts with current market price for market-based metrics",
            "",
            "Latest indicator observations:",
            *observation_lines,
        ]
        documents.append(
            Document(
                text="\n".join(lines),
                metadata={
                    "type": "stock_indicator_snapshot",
                    "ticker": ticker,
                    "company_name": company_profile["company_name"],
                    "sector": company_profile["sector"],
                    "industry": company_profile["industry"],
                    "group": "company",
                    "subgroup": "financial_indicators",
                    "glossary_domain": "company",
                    "frequency": frequency,
                    "period_end_date": period_end_date,
                    "source": "SEC Company Facts with current market price for market-based metrics",
                },
            )
        )

    return documents


def _macro_observation_documents(db_path, glossary_lookup):
    latest_rows = _latest_macro_rows(db_path)
    if latest_rows.empty:
        return []

    observation_lines = []
    for _, row in latest_rows.iterrows():
        indicator_metadata = _lookup_indicator_metadata(
            row["indicator_name"],
            glossary_lookup,
            fallback_group="macro",
            fallback_subgroup=row.get("category") or "macro",
            fallback_domain="eco",
        )
        observation_lines.append(
            (
                f"Indicator: {row['indicator_name']} | Value: {row['value']} {row['units']} "
                f"| Observation Date: {row['observation_date'].strftime('%Y-%m-%d')} "
                f"| Group: {indicator_metadata.get('group') or 'macro'} "
                f"| Subgroup: {indicator_metadata.get('subgroup') or row.get('category') or 'macro'} "
                f"| Release: {row['release_name']}"
            )
        )

    if not observation_lines:
        return []

    return [
        Document(
            text="\n".join(
                [
                    "Observation Type: Latest macro indicator snapshot",
                    "Source: Macro SQL database",
                    "",
                    "Latest macro observations:",
                    *observation_lines,
                ]
            ),
            metadata={
                "type": "macro_indicator_snapshot",
                "group": "macro",
                "subgroup": "macro",
                "glossary_domain": "eco",
                "source": "Macro SQL database",
            },
        )
    ]


def _dedupe_documents(documents):
    deduped_documents = []
    seen_keys = set()
    for document in documents:
        metadata = document.metadata or {}
        key = (
            json.dumps(metadata, sort_keys=True, default=str),
            (document.text or "").strip(),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_documents.append(document)
    return deduped_documents


def build_graph_documents_for_ticker(
    ticker,
    stock_docs=None,
    stock_db_path=ingest_stock.DEFAULT_STOCK_DB_PATH,
    macro_db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
    filings_base_dir=ingest_stock.DEFAULT_STOCK_FILINGS_BASE_DIR,
    glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
):
    ticker = ticker.upper()
    stock_docs = stock_docs or ingest_stock.build_financial_docs(
        ticker,
        db_path=stock_db_path,
        filings_base_dir=filings_base_dir,
    )
    stock_docs = _select_stock_docs_for_graph(stock_docs)
    knowledge_docs = ingest_knowledge.build_glossary_docs(
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    macro_docs = ingest_macro.build_market_environment_docs(db_path=macro_db_path)

    glossary_lookup = _glossary_indicator_lookup(knowledge_docs)
    stock_observation_docs = _stock_observation_documents(
        ticker,
        db_path=stock_db_path,
        glossary_lookup=glossary_lookup,
        stock_docs=stock_docs,
    )
    macro_observation_docs = _macro_observation_documents(
        db_path=macro_db_path,
        glossary_lookup=glossary_lookup,
    )

    prepared_documents = []
    for doc in knowledge_docs:
        prepared_documents.append(_prepare_graph_document(doc, "knowledge_glossary"))
    for doc in macro_docs:
        prepared_documents.append(_prepare_graph_document(doc, "macro_context"))
    for doc in stock_docs:
        prepared_documents.append(_prepare_graph_document(doc, "stock_context"))
    for doc in stock_observation_docs:
        prepared_documents.append(_prepare_graph_document(doc, "stock_observation"))
    for doc in macro_observation_docs:
        prepared_documents.append(_prepare_graph_document(doc, "macro_observation"))

    return _dedupe_documents(prepared_documents)


def _graph_extractors():
    global _GRAPH_LLM
    if _GRAPH_LLM is None:
        _GRAPH_LLM = OpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=0)

    return [
        DynamicLLMPathExtractor(
            llm=_GRAPH_LLM,
            max_triplets_per_chunk=4,
            num_workers=1,
            allowed_entity_types=GRAPH_ENTITY_TYPES,
            allowed_relation_types=GRAPH_RELATION_TYPES,
            allowed_entity_props=GRAPH_ENTITY_PROPS,
            allowed_relation_props=GRAPH_RELATION_PROPS,
        ),
        ImplicitPathExtractor(),
    ]


def refresh_property_graph_for_ticker(
    ticker,
    stock_docs=None,
    stock_db_path=ingest_stock.DEFAULT_STOCK_DB_PATH,
    macro_db_path=ingest_macro.DEFAULT_MACRO_DB_PATH,
    filings_base_dir=ingest_stock.DEFAULT_STOCK_FILINGS_BASE_DIR,
    glossary_base_dir=ingest_knowledge.DEFAULT_GLOSSARY_BASE_DIR,
    metadata_path=ingest_knowledge.DEFAULT_GLOSSARY_METADATA_PATH,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    ingest_stock.env()
    ingest_macro.refresh_market_environment_if_stale(db_path=macro_db_path)

    documents = build_graph_documents_for_ticker(
        ticker,
        stock_docs=stock_docs,
        stock_db_path=stock_db_path,
        macro_db_path=macro_db_path,
        filings_base_dir=filings_base_dir,
        glossary_base_dir=glossary_base_dir,
        metadata_path=metadata_path,
    )
    if not documents:
        return None

    persist_dir = _graph_persist_dir(ticker, storage_dir=storage_dir)
    _reset_persist_dir(persist_dir)

    try:
        storage_context = StorageContext.from_defaults(
            property_graph_store=SimplePropertyGraphStore()
        )
        PropertyGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[SentenceSplitter(chunk_size=4096, chunk_overlap=150)],
            kg_extractors=_graph_extractors(),
            embed_kg_nodes=False,
            show_progress=False,
            use_async=False,
        )
        storage_context.persist(
            persist_dir=persist_dir,
            fs=_UTF8LocalFileSystem(auto_mkdir=True),
        )
    except Exception as exc:
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir, onerror=_remove_readonly)
        raise RuntimeError(f"Property graph refresh failed for {ticker}: {exc}") from exc

    return persist_dir


def refresh_full_graph_for_ticker(*args, **kwargs):
    return refresh_property_graph_for_ticker(*args, **kwargs)


def _load_graph_index(ticker, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    persist_dir = _graph_persist_dir(ticker, storage_dir=storage_dir)
    if not graph_index_exists(ticker, storage_dir=storage_dir):
        raise FileNotFoundError(f"Graph index directory not found: {persist_dir}")

    ingest_stock.env()
    fs = _UTF8LocalFileSystem(auto_mkdir=True)
    property_graph_store = SimplePropertyGraphStore.from_persist_dir(persist_dir, fs=fs)
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        property_graph_store=property_graph_store,
        fs=fs,
    )
    return load_index_from_storage(storage_context)


def _node_text(node_with_score):
    node = getattr(node_with_score, "node", node_with_score)
    if hasattr(node, "get_content"):
        text = node.get_content()
    else:
        text = getattr(node, "text", "")
    return str(text).replace("\ufeff", "").strip()


def retrieve_graph_context(
    query_str,
    ticker,
    storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
    similarity_top_k=4,
    max_chunks=4,
):
    try:
        index = _load_graph_index(ticker, storage_dir=storage_dir)
    except (FileNotFoundError, ValueError):
        return None

    retriever = index.as_retriever(
        include_text=True,
        similarity_top_k=similarity_top_k,
        path_depth=1,
    )
    nodes = retriever.retrieve(query_str)
    chunks = []
    seen_chunks = set()

    for node in nodes:
        text = _node_text(node)
        if not text:
            continue
        if text in seen_chunks:
            continue
        seen_chunks.add(text)
        chunks.append(text)
        if len(chunks) >= max_chunks:
            break

    if not chunks:
        return None

    return "Graph property context:\n" + "\n\n".join(chunks)
