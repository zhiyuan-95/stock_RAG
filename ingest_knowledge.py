import os
import shutil
import stat
import subprocess

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import ingest_stock


load_dotenv("config.env")

DEFAULT_GLOSSARY_BASE_DIR = os.getenv("GLOSSARY_BASE_DIR", "./data_store/glossary")
DEFAULT_KNOWLEDGE_STORAGE_DIR = os.getenv("KNOWLEDGE_STORAGE_DIR", "./storage/knowledge")


def _clean_text_field(value):
    if value is None:
        return None
    cleaned_value = str(value).strip()
    return cleaned_value or None


def _indicator_name_from_path(glossary_path):
    indicator_name = os.path.splitext(os.path.basename(glossary_path))[0]
    return indicator_name.replace("_", " ").strip()


def build_glossary_docs(glossary_base_dir=DEFAULT_GLOSSARY_BASE_DIR):
    glossary_dir = os.path.join(glossary_base_dir, "specific_indicators")
    if not os.path.isdir(glossary_dir):
        return []

    glossary_documents = []
    for entry in sorted(os.scandir(glossary_dir), key=lambda item: item.name.lower()):
        if not entry.is_file() or not entry.name.lower().endswith(".md"):
            continue

        with open(entry.path, "r", encoding="utf-8") as glossary_file:
            text = glossary_file.read().strip()

        if not text:
            continue

        indicator_name = None
        for line in text.splitlines():
            if line.lower().startswith("indicator:"):
                indicator_name = _clean_text_field(line.split(":", 1)[1])
                break

        glossary_documents.append(
            Document(
                text=text,
                metadata={
                    "type": "indicator_glossary",
                    "indicator_name": indicator_name or _indicator_name_from_path(entry.path),
                    "glossary_category": "specific_indicator",
                    "glossary_path": entry.path,
                    "source": "glossary_markdown",
                },
            )
        )

    return glossary_documents


def _clear_windows_readonly(path):
    if os.name != "nt" or not os.path.lexists(path):
        return
    subprocess.run(
        ["cmd", "/c", "attrib", "-R", path],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
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


def refresh_knowledge_index(
    glossary_base_dir=DEFAULT_GLOSSARY_BASE_DIR,
    storage_dir=DEFAULT_KNOWLEDGE_STORAGE_DIR,
):
    ingest_stock.env()

    docs = build_glossary_docs(glossary_base_dir=glossary_base_dir)
    if not docs:
        print("No glossary documents found; skipping knowledge index refresh.")
        return None

    os.makedirs(os.path.dirname(os.path.normpath(storage_dir)) or ".", exist_ok=True)
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = node_parser.get_nodes_from_documents(docs)
    _reset_persist_dir(storage_dir)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=storage_dir)
    print(f"Knowledge index successfully refreshed at {storage_dir}")
    return storage_dir
