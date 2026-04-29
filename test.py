import argparse
import json
import os
import re
import textwrap
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import ingest_news


NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
NEWSAPI_SOURCE_IDS = (
    "bloomberg",
    "reuters",
    "financial-times",
    "the-wall-street-journal",
    "the-economist",
    "associated-press",
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
DEFAULT_RECENT_NEWS_QUERY = (
    "market OR economy OR business OR company OR government OR policy OR "
    "technology OR stocks OR earnings OR inflation OR rates OR bonds OR "
    "commodities OR sports OR entertainment OR health OR weather OR politics"
)
BAND_ORDER = ("band_A", "band_B", "band_C", "band_f")
BAND_OUTPUT_FILES = {
    "band_A": "band_A",
    "band_B": "band_B",
    "band_C": "band_C",
    "band_f": "band_f",
}
BAND_TO_SCORE_BAND = {
    "band_a": "band_A",
    "a": "band_A",
    "band_b": "band_B",
    "b": "band_B",
    "band_c": "drop",
    "c": "drop",
    "drop": "drop",
    "dropped": "drop",
    "band_f": "band_f",
    "f": "band_f",
}
SCORE_BAND_TO_REPORT_BAND = {
    "band_A": "band_A",
    "band_B": "band_B",
    "drop": "band_C",
    "band_f": "band_f",
}


def log_step(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def load_config_env(path="config.env"):
    values = {}
    if not os.path.exists(path):
        return values

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip().upper()] = value.strip().strip("\"'")
    return values


def get_config_value(name, aliases=()):
    config = load_config_env()
    for key in (name, *aliases):
        value = os.getenv(key) or config.get(key.upper())
        if str(value or "").strip():
            return str(value).strip()
    return None


def get_newsapi_key():
    api_key = get_config_value("NEWSAPI_KEY", aliases=("NEWS_API_KEY",))
    if not api_key:
        raise ValueError("NewsAPI key not found. Add NEWSAPI_KEY=your_key to config.env.")
    return api_key


def clean_newsapi_content_marker(value):
    text = str(value or "").strip()
    if not text:
        return None
    return re.sub(r"\s*(?:…|\.{3})?\s*\[\+\d+\s+chars\]\s*$", "", text).strip() or None


class LlamaIndexLLMClient:
    def __init__(self):
        log_step("Loading LLM dependencies and project environment...")
        try:
            import ingest_stock
            from llama_index.core import Settings
        except ImportError as exc:
            raise RuntimeError(
                "Prompt A scoring needs your project LLM dependencies. "
                "Run this with the same Python environment that can run ingest_stock.env()."
            ) from exc

        ingest_stock.env()
        self.settings = Settings
        log_step("LLM client is ready.")

    def complete(self, prompt):
        return str(self.settings.llm.complete(prompt))


class ProgressLLMClient:
    def __init__(self, llm_client, total_batches):
        self.llm_client = llm_client
        self.total_batches = total_batches
        self.completed_batches = 0

    def complete(self, prompt):
        batch_number = self.completed_batches + 1
        log_step(f"Scoring batch {batch_number}/{self.total_batches} with Prompt A...")
        response = self.llm_client.complete(prompt)
        self.completed_batches += 1
        log_step(f"Finished batch {self.completed_batches}/{self.total_batches}.")
        return response


def fetch_recent_newsapi_articles(
    limit=100,
    sources=NEWSAPI_SOURCE_IDS,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
):
    page_size = max(1, min(int(limit), 100))
    log_step(f"Checking NewsAPI key and preparing request for {page_size} recent articles...")
    params = {
        "apiKey": get_newsapi_key(),
        "sources": ",".join(sources),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    if search_query:
        params["q"] = search_query

    request = Request(
        f"{NEWSAPI_EVERYTHING_URL}?{urlencode(params)}",
        headers={"User-Agent": "Stock-RAG-score-debug/1.0"},
    )

    log_step("Calling NewsAPI /v2/everything...")
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("status") != "ok":
        message = payload.get("message") or payload
        raise RuntimeError(f"NewsAPI request failed: {message}")

    articles = payload.get("articles", [])
    sorted_articles = sorted(
        articles,
        key=lambda article: article.get("publishedAt") or "",
        reverse=True,
    )[:page_size]
    log_step(f"Fetched {len(sorted_articles)} articles from NewsAPI.")
    return sorted_articles


def to_raw_article(raw_article):
    source = raw_article.get("source") or {}
    return ingest_news.RawArticle.model_validate(
        {
            "published_at": raw_article.get("publishedAt"),
            "source_name": source.get("name"),
            "title": raw_article.get("title"),
            "url": raw_article.get("url"),
            "description": raw_article.get("description"),
            "content": clean_newsapi_content_marker(raw_article.get("content")),
        }
    )


def normalize_requested_band(value):
    if value is None or str(value).lower() in {"all", "all_bands"}:
        return None

    normalized = BAND_TO_SCORE_BAND.get(str(value).strip().lower())
    if normalized is None:
        allowed = ", ".join(BAND_ORDER)
        raise ValueError(f"Unknown band '{value}'. Use one of: {allowed}, all")
    return normalized


def wrap_text(label, value, width=110, indent="    "):
    text = str(value or "").strip() or "None"
    wrapped = textwrap.fill(
        text,
        width=width,
        initial_indent=f"{label}: ",
        subsequent_indent=indent,
    )
    return wrapped


def group_scored_articles(raw_articles, score_results):
    grouped = {band: [] for band in BAND_ORDER}
    for index, article in enumerate(raw_articles, start=1):
        score_result = score_results.get(article.article_id)
        if score_result is None:
            report_band = "band_C"
        else:
            report_band = SCORE_BAND_TO_REPORT_BAND[score_result.score_band]
        grouped[report_band].append((index, article, score_result))
    return grouped


def format_article_report(index, article, score_result):
    score = "missing" if score_result is None else score_result.score
    score_band = "missing" if score_result is None else score_result.score_band
    report_band = "band_C" if score_band in {"drop", "missing"} else score_band
    reason = None
    negative_text = None
    if score_result is not None:
        reason = score_result.market_relevance_reason
        negative_text = score_result.extracted_negative_text

    lines = [
        "-" * 120,
        f"#{index:03d} | score={score} | program_band={score_band} | report_band={report_band}",
        f"Source: {article.source_name or 'Unknown'}",
        f"Published: {article.published_at or 'Unknown'}",
        f"URL: {article.url or 'None'}",
        wrap_text("Title", article.title),
    ]
    if score_band == "band_f":
        lines.append(wrap_text("Extracted negative text", negative_text))
    else:
        lines.append(wrap_text("Market relevance reason", reason))
    lines.extend(
        [
            "",
            "DESCRIPTION:",
            str(article.description or "None").strip() or "None",
            "",
            "FULL CONTENT FROM NEWSAPI:",
            str(article.content or "None").strip() or "None",
            "",
        ]
    )
    return "\n".join(lines)


def print_article(index, article, score_result):
    print(format_article_report(index, article, score_result))


def print_band_report(grouped, requested_score_band=None):
    total = sum(len(items) for items in grouped.values())
    print("=" * 120)
    print("NEWSAPI PROMPT A SCORE DEBUG REPORT")
    print("Read-only check: NewsAPI fetch + Prompt A scoring only. No ingestion is performed.")
    print("band_C is the report label for program score_band='drop'.")
    print(f"Total scored articles: {total}")
    print(
        "Counts: "
        + " | ".join(f"{band}={len(grouped[band])}" for band in BAND_ORDER)
    )
    print("=" * 120)
    print()

    for report_band in BAND_ORDER:
        score_band = BAND_TO_SCORE_BAND[report_band.lower()]
        if requested_score_band is not None and requested_score_band != score_band:
            continue

        print()
        print("#" * 120)
        print(f"{report_band} ({len(grouped[report_band])} articles)")
        if report_band == "band_C":
            print("Program score_band: drop")
        print("#" * 120)
        print()

        if not grouped[report_band]:
            print("No articles in this band.")
            continue

        for index, article, score_result in grouped[report_band]:
            print_article(index, article, score_result)


def build_band_file_text(report_band, grouped, generated_at):
    total = sum(len(items) for items in grouped.values())
    lines = [
        "NEWSAPI PROMPT A SCORE DEBUG REPORT",
        f"Generated at: {generated_at}",
        "Read-only check: NewsAPI fetch + Prompt A scoring only. No ingestion was performed.",
        "band_C is the report label for program score_band='drop'.",
        f"Current file: {report_band}",
        f"Articles in this file: {len(grouped[report_band])}",
        f"Total scored articles: {total}",
        "Counts: " + " | ".join(f"{band}={len(grouped[band])}" for band in BAND_ORDER),
        "=" * 120,
        "",
    ]

    if report_band == "band_C":
        lines.extend(["Program score_band: drop", ""])

    if not grouped[report_band]:
        lines.append("No articles in this band.")
    else:
        for index, article, score_result in grouped[report_band]:
            lines.append(format_article_report(index, article, score_result))

    lines.append("")
    return "\n".join(lines)


def write_band_report_files(grouped, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    written_paths = {}

    for report_band in BAND_ORDER:
        path = os.path.join(output_dir, BAND_OUTPUT_FILES[report_band])
        file_text = build_band_file_text(report_band, grouped, generated_at)
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(file_text)
        written_paths[report_band] = path
        log_step(f"Wrote {len(grouped[report_band])} articles to {path}")

    return written_paths


def generate_band_report_files(
    limit=100,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    log_step("Starting read-only news scoring check. No ingestion will be performed.")
    newsapi_articles = fetch_recent_newsapi_articles(limit=limit, search_query=search_query)
    log_step("Converting NewsAPI payloads into ingest_news.RawArticle objects...")
    raw_articles = [to_raw_article(article) for article in newsapi_articles]
    log_step(f"Prepared {len(raw_articles)} RawArticle objects.")

    total_batches = (len(raw_articles) + ingest_news.NEWS_BATCH_SIZE - 1) // ingest_news.NEWS_BATCH_SIZE
    llm_client = ProgressLLMClient(
        LlamaIndexLLMClient(),
        total_batches=total_batches,
    )

    log_step(
        f"Running Prompt A scoring for {len(raw_articles)} articles "
        f"in {total_batches} batch(es)..."
    )
    score_results = ingest_news.run_prompt_a_scoring(raw_articles, llm_client)
    log_step(f"Prompt A returned scores for {len(score_results)} articles.")

    log_step("Grouping scored articles into band_A, band_B, band_C/drop, and band_f...")
    grouped = group_scored_articles(raw_articles, score_results)
    log_step(f"Writing band files into {output_dir}...")
    written_paths = write_band_report_files(grouped, output_dir=output_dir)
    log_step("Done.")
    return written_paths


def score_recent_news(limit=100, requested_band=None, search_query=DEFAULT_RECENT_NEWS_QUERY):
    log_step("Starting read-only news scoring check. No ingestion will be performed.")
    newsapi_articles = fetch_recent_newsapi_articles(limit=limit, search_query=search_query)
    log_step("Converting NewsAPI payloads into ingest_news.RawArticle objects...")
    raw_articles = [to_raw_article(article) for article in newsapi_articles]
    log_step(f"Prepared {len(raw_articles)} RawArticle objects.")

    total_batches = (len(raw_articles) + ingest_news.NEWS_BATCH_SIZE - 1) // ingest_news.NEWS_BATCH_SIZE
    llm_client = ProgressLLMClient(
        LlamaIndexLLMClient(),
        total_batches=total_batches,
    )

    log_step(
        f"Running Prompt A scoring for {len(raw_articles)} articles "
        f"in {total_batches} batch(es)..."
    )
    score_results = ingest_news.run_prompt_a_scoring(raw_articles, llm_client)
    log_step(f"Prompt A returned scores for {len(score_results)} articles.")

    log_step("Grouping scored articles into band_A, band_B, band_C/drop, and band_f...")
    grouped = group_scored_articles(raw_articles, score_results)
    log_step("Printing readable band report...")
    print_band_report(grouped, requested_score_band=requested_band)
    log_step("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent NewsAPI articles, run Prompt A scoring only, and write "
            "four readable band files without ingestion."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of recent NewsAPI articles to score. Max 100.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_RECENT_NEWS_QUERY,
        help="NewsAPI /v2/everything query. Use a broad query to inspect dropped news.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where band_A, band_B, band_C, and band_f files will be written.",
    )
    args = parser.parse_args()

    written_paths = generate_band_report_files(
        limit=args.limit,
        search_query=args.query,
        output_dir=args.output_dir,
    )
    print()
    print("Generated files:")
    for report_band in BAND_ORDER:
        print(f"{report_band}: {written_paths[report_band]}")


if __name__ == "__main__":
    main()
