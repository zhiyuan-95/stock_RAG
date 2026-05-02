import argparse
import json
import os
import re
import socket
import textwrap
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import ingest_news


WORLD_NEWS_SEARCH_URL = "https://api.worldnewsapi.com/search-news"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_PROMPT_A_MODEL = "gpt-4.1-mini"
DEFAULT_PROMPT_A_BATCH_SIZE = 5
DEFAULT_PROMPT_A_TIMEOUT_SECONDS = 300
DEFAULT_PROMPT_A_RETRIES = 3
DEFAULT_PROMPT_B_BATCH_SIZE = 3
DEFAULT_FAST_FILTER_EMBED_MODEL = ingest_news.DEFAULT_NEGATIVE_FILTER_EMBED_MODEL
DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD = 0.85
WORLD_NEWS_SOURCE_URLS = (
    "https://www.bloomberg.com",
    "https://www.reuters.com",
    "https://www.ft.com",
    "https://www.wsj.com",
    "https://www.economist.com",
    "https://apnews.com",
)
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
DEFAULT_RECENT_NEWS_QUERY = None
SOURCE_NAME_BY_DOMAIN = {
    "bloomberg.com": "Bloomberg",
    "reuters.com": "Reuters",
    "ft.com": "Financial Times",
    "wsj.com": "The Wall Street Journal",
    "economist.com": "The Economist",
    "apnews.com": "Associated Press",
}
FAST_FILTER_REPORT = "fast_filtered"
BAND_ORDER = ("band_A", "band_B", "band_C", "band_f")
REPORT_ORDER = (FAST_FILTER_REPORT, *BAND_ORDER)
BAND_OUTPUT_FILES = {
    FAST_FILTER_REPORT: FAST_FILTER_REPORT,
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
    "band_d": "drop",
    "c": "drop",
    "d": "drop",
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


def get_world_news_api_key():
    api_key = get_config_value(
        "WORLD_NEWS_API_KEY",
        aliases=("World_News_API_KEY", "WORLD_NEWS_KEY"),
    )
    if not api_key:
        raise ValueError(
            "World News API key not found. Add World_News_API_KEY=your_key to config.env."
        )
    return api_key


def get_openai_api_key():
    api_key = get_config_value("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Add OPENAI_API_KEY=your_key to config.env."
        )
    return api_key


def normalize_prompt_a_model(model):
    normalized = re.sub(r"\s+", "-", str(model or "").strip().lower())
    aliases = {
        "4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt4.1-mini": "gpt-4.1-mini",
        "gpt-4-1-mini": "gpt-4.1-mini",
        "gemini-4.1-mini": "gpt-4.1-mini",
        "gemini-4-1-mini": "gpt-4.1-mini",
    }
    return aliases.get(normalized, normalized)


def is_openai_model(model):
    return model.startswith(("gpt-", "o1", "o3", "o4", "o5"))


def is_gemini_model(model):
    return model.startswith("gemini-")


def clean_world_news_text(value):
    text = str(value or "").strip()
    return text or None


def summary_from_world_news_article(raw_article, max_chars=900):
    summary = clean_world_news_text(
        raw_article.get("summary")
        or raw_article.get("description")
        or raw_article.get("excerpt")
    )
    if summary:
        return summary

    text = clean_world_news_text(raw_article.get("text"))
    if not text:
        return None

    normalized = re.sub(r"\s+", " ", text).strip()
    sentence_matches = re.findall(r".*?(?:[.!?](?=\s+[A-Z0-9\"'])|$)", normalized)
    sentences = [sentence.strip() for sentence in sentence_matches if sentence.strip()]
    if not sentences:
        return normalized[:max_chars].rstrip()

    selected = []
    current_length = 0
    for sentence in sentences:
        next_length = current_length + len(sentence) + (1 if selected else 0)
        if selected and next_length > max_chars:
            break
        selected.append(sentence)
        current_length = next_length
        if len(selected) >= 3:
            break

    return " ".join(selected).strip() or normalized[:max_chars].rstrip()


def source_name_from_url(url):
    host = str(url or "").lower()
    for domain, source_name in SOURCE_NAME_BY_DOMAIN.items():
        if domain in host:
            return source_name
    return None


def normalize_anchor_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


class SentenceTransformerEmbeddingClient:
    def __init__(self, model=DEFAULT_FAST_FILTER_EMBED_MODEL, batch_size=64):
        self.model = model
        self.batch_size = batch_size
        log_step(f"Loading fast-filter SentenceTransformer embedding model: {self.model}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the fast filter. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self.encoder = SentenceTransformer(self.model)
        log_step(f"Fast filter will use local embedding model: {self.model}")

    def embed_texts(self, texts):
        texts = [str(text or " ") for text in texts]
        if not texts:
            return []

        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for batch_number, offset in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[offset : offset + self.batch_size]
            log_step(
                f"Encoding fast-filter embeddings batch {batch_number}/{total_batches} "
                f"({len(batch)} item(s)) for fast filter..."
            )
            batch_embeddings = self.encoder.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings.extend(vector.tolist() for vector in batch_embeddings)

        return embeddings


class OpenAIChatLLMClient:
    def __init__(
        self,
        model,
        timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
        max_retries=DEFAULT_PROMPT_A_RETRIES,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, int(max_retries))
        self.api_key = get_openai_api_key()
        log_step(
            f"Prompt A will use OpenAI model: {self.model} "
            f"(timeout={self.timeout_seconds}s, retries={self.max_retries})"
        )

    def complete(self, prompt):
        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }

        payload = None
        for attempt in range(1, self.max_retries + 1):
            request = Request(
                OPENAI_CHAT_COMPLETIONS_URL,
                data=json.dumps(request_body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                error_payload = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI Prompt A request failed: {error_payload}") from exc
                sleep_seconds = min(20, 2 * attempt)
                log_step(
                    f"OpenAI Prompt A request failed with HTTP {exc.code}; "
                    f"retrying in {sleep_seconds}s ({attempt}/{self.max_retries})..."
                )
                time.sleep(sleep_seconds)
            except (TimeoutError, socket.timeout, URLError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        "OpenAI Prompt A request timed out or lost connection after "
                        f"{self.max_retries} attempt(s)."
                    ) from exc
                sleep_seconds = min(20, 2 * attempt)
                log_step(
                    "OpenAI Prompt A request timed out or lost connection; "
                    f"retrying in {sleep_seconds}s ({attempt}/{self.max_retries})..."
                )
                time.sleep(sleep_seconds)

        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenAI Prompt A response: {payload}") from exc


class GeminiLlamaIndexLLMClient:
    def __init__(self, model):
        self.model = model
        log_step("Loading Gemini LLM dependencies and project environment...")
        try:
            import ingest_stock
            from llama_index.core import Settings
        except ImportError as exc:
            raise RuntimeError(
                "Gemini Prompt A scoring needs your project LLM dependencies. "
                "Run this with the same Python environment that can run ingest_stock.env()."
            ) from exc

        ingest_stock.env(llm_model=self.model)
        self.settings = Settings
        log_step(f"Prompt A will use Gemini model: {self.model}")

    def complete(self, prompt):
        return str(self.settings.llm.complete(prompt))


class LlamaIndexLLMClient:
    def __init__(
        self,
        model=DEFAULT_PROMPT_A_MODEL,
        timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
        max_retries=DEFAULT_PROMPT_A_RETRIES,
    ):
        self.model = normalize_prompt_a_model(model)
        if is_openai_model(self.model):
            self.client = OpenAIChatLLMClient(
                self.model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        elif is_gemini_model(self.model):
            self.client = GeminiLlamaIndexLLMClient(self.model)
        else:
            raise ValueError(
                f"Unsupported Prompt A model '{model}'. Use gpt-4.1-mini or a Gemini "
                "model ID like gemini-2.5-flash."
            )
        log_step("LLM client is ready.")

    def complete(self, prompt):
        return self.client.complete(prompt)


class ProgressLLMClient:
    def __init__(self, llm_client, total_batches, task_name="Prompt A", action="Scoring"):
        self.llm_client = llm_client
        self.total_batches = total_batches
        self.completed_batches = 0
        self.task_name = task_name
        self.action = action

    def complete(self, prompt):
        batch_number = self.completed_batches + 1
        log_step(f"{self.action} batch {batch_number}/{self.total_batches} with {self.task_name}...")
        response = self.llm_client.complete(prompt)
        self.completed_batches += 1
        log_step(f"Finished batch {self.completed_batches}/{self.total_batches}.")
        return response


def fetch_recent_world_news_articles(
    limit=100,
    sources=WORLD_NEWS_SOURCE_URLS,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
):
    page_size = max(1, min(int(limit), 100))
    log_step(f"Checking World News API key and preparing request for {page_size} recent articles...")
    params = {
        "news-sources": ",".join(sources),
        "language": "en",
        "sort": "publish-time",
        "sort-direction": "DESC",
        "number": page_size,
    }
    if search_query:
        if len(search_query) > 100:
            raise ValueError("World News API text query must be 100 characters or fewer.")
        params["text"] = search_query
        params["text-match-indexes"] = "title,content"

    request = Request(
        f"{WORLD_NEWS_SEARCH_URL}?{urlencode(params)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "Stock-RAG-world-news-score-debug/1.0",
            "x-api-key": get_world_news_api_key(),
        },
    )

    log_step("Calling World News API /search-news...")
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "news" not in payload:
        message = payload.get("message") or payload
        raise RuntimeError(f"World News API request failed: {message}")

    articles = payload.get("news", [])
    sorted_articles = sorted(
        articles,
        key=lambda article: article.get("publish_date") or "",
        reverse=True,
    )[:page_size]
    log_step(f"Fetched {len(sorted_articles)} articles from World News API.")
    return sorted_articles


def to_raw_article(raw_article):
    return ingest_news.RawArticle.model_validate(
        {
            "published_at": raw_article.get("publish_date"),
            "source_name": source_name_from_url(raw_article.get("url")),
            "title": raw_article.get("title"),
            "url": raw_article.get("url"),
            "description": summary_from_world_news_article(raw_article),
            "content": clean_world_news_text(raw_article.get("text")),
        }
    )


def run_fast_negative_filter(
    raw_articles,
    threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
    embed_model=DEFAULT_FAST_FILTER_EMBED_MODEL,
):
    log_step(
        f"Building fast negative-anchor filter with threshold {threshold:.2f}..."
    )
    embedding_client = SentenceTransformerEmbeddingClient(model=embed_model)
    negative_texts = [
        normalize_anchor_text(text)
        for text in ingest_news.NEGATIVE_TEXTS
        if normalize_anchor_text(text)
    ]

    log_step(f"Embedding {len(negative_texts)} negative anchor text(s)...")
    negative_anchor_vectors = embedding_client.embed_texts(negative_texts)

    article_texts = [
        normalize_anchor_text(article.fast_filter_text())
        for article in raw_articles
    ]
    non_empty_positions = [
        index for index, text in enumerate(article_texts) if text
    ]
    non_empty_texts = [article_texts[index] for index in non_empty_positions]

    log_step(
        f"Embedding {len(non_empty_texts)} article title + first-description-sentence text(s)..."
    )
    non_empty_vectors = embedding_client.embed_texts(non_empty_texts)
    vector_by_position = dict(zip(non_empty_positions, non_empty_vectors))

    log_step("Calculating maximum cosine similarity for each article...")
    passed_articles = []
    decisions = []
    for position, article in enumerate(raw_articles):
        vector = vector_by_position.get(position)
        if vector is None or not negative_anchor_vectors:
            decision = ingest_news.NegativeAnchorDecision(
                article_id=article.article_id,
                passed=True,
                max_similarity=0.0,
                matched_negative_text=None,
            )
        else:
            similarities = [
                ingest_news.cosine_similarity(vector, anchor_vector)
                for anchor_vector in negative_anchor_vectors
            ]
            max_similarity = max(similarities) if similarities else 0.0
            match_index = similarities.index(max_similarity) if similarities else -1
            matched_text = negative_texts[match_index] if match_index >= 0 else None
            decision = ingest_news.NegativeAnchorDecision(
                article_id=article.article_id,
                passed=max_similarity < threshold,
                max_similarity=max_similarity,
                matched_negative_text=matched_text,
            )

        decisions.append(decision)
        if decision.passed:
            passed_articles.append(article)

    filtered_count = len(raw_articles) - len(passed_articles)
    log_step(
        f"Fast filter removed {filtered_count} article(s) and passed "
        f"{len(passed_articles)} article(s) to Prompt A."
    )
    return passed_articles, decisions


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


def group_scored_articles_with_original_indexes(raw_articles, score_results, original_indexes):
    grouped = {band: [] for band in BAND_ORDER}
    for position, article in enumerate(raw_articles, start=1):
        index = original_indexes.get(article.article_id, position)
        score_result = score_results.get(article.article_id)
        if score_result is None:
            report_band = "band_C"
        else:
            report_band = SCORE_BAND_TO_REPORT_BAND[score_result.score_band]
        grouped[report_band].append((index, article, score_result))
    return grouped


def build_fast_filtered_items(raw_articles, filter_decisions):
    articles_by_id = {article.article_id: article for article in raw_articles}
    original_indexes = {
        article.article_id: index for index, article in enumerate(raw_articles, start=1)
    }
    return [
        (original_indexes[decision.article_id], articles_by_id[decision.article_id], decision)
        for decision in filter_decisions
        if not decision.passed and decision.article_id in articles_by_id
    ]


def print_distribution_summary(grouped, fast_filtered_items):
    total_prompt_a = sum(len(items) for items in grouped.values())
    total_articles = total_prompt_a + len(fast_filtered_items)
    print()
    print("=" * 120, flush=True)
    print("ARTICLE DISTRIBUTION BEFORE WRITING FILES", flush=True)
    print(f"Total fetched articles: {total_articles}", flush=True)
    print(f"{FAST_FILTER_REPORT}: {len(fast_filtered_items)}", flush=True)
    for report_band in BAND_ORDER:
        print(f"{report_band}: {len(grouped[report_band])}", flush=True)
    print("=" * 120, flush=True)
    print(flush=True)


def run_read_only_scoring_pipeline(
    limit=100,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
    prompt_a_model=DEFAULT_PROMPT_A_MODEL,
    prompt_a_batch_size=DEFAULT_PROMPT_A_BATCH_SIZE,
    prompt_a_timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
    prompt_a_retries=DEFAULT_PROMPT_A_RETRIES,
    fast_filter_embed_model=DEFAULT_FAST_FILTER_EMBED_MODEL,
    negative_similarity_threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    log_step("Starting read-only news scoring check. No ingestion will be performed.")
    world_news_articles = fetch_recent_world_news_articles(limit=limit, search_query=search_query)
    log_step("Converting World News API payloads into ingest_news.RawArticle objects...")
    raw_articles = [to_raw_article(article) for article in world_news_articles]
    log_step(f"Prepared {len(raw_articles)} RawArticle objects.")

    original_indexes = {
        article.article_id: index for index, article in enumerate(raw_articles, start=1)
    }
    passed_articles, filter_decisions = run_fast_negative_filter(
        raw_articles,
        threshold=negative_similarity_threshold,
        embed_model=fast_filter_embed_model,
    )
    fast_filtered_items = build_fast_filtered_items(raw_articles, filter_decisions)

    prompt_a_batch_size = max(1, int(prompt_a_batch_size))
    total_batches = (len(passed_articles) + prompt_a_batch_size - 1) // prompt_a_batch_size
    if passed_articles:
        llm_client = ProgressLLMClient(
            LlamaIndexLLMClient(
                prompt_a_model,
                timeout_seconds=prompt_a_timeout_seconds,
                max_retries=prompt_a_retries,
            ),
            total_batches=total_batches,
        )
        log_step(
            f"Running Prompt A scoring for {len(passed_articles)} fast-filter-passed articles "
            f"in {total_batches} batch(es) of up to {prompt_a_batch_size}..."
        )
        score_results = ingest_news.run_prompt_a_scoring(
            passed_articles,
            llm_client,
            batch_size=prompt_a_batch_size,
        )
    else:
        log_step("No articles passed the fast filter; skipping Prompt A scoring.")
        score_results = {}
    log_step(f"Prompt A returned scores for {len(score_results)} articles.")

    log_step("Grouping scored articles into band_A, band_B, band_C/drop, and band_f...")
    grouped = group_scored_articles_with_original_indexes(
        passed_articles,
        score_results,
        original_indexes,
    )
    return {
        "world_news_articles": world_news_articles,
        "raw_articles": raw_articles,
        "passed_articles": passed_articles,
        "filter_decisions": filter_decisions,
        "fast_filtered_items": fast_filtered_items,
        "score_results": score_results,
        "grouped": grouped,
        "original_indexes": original_indexes,
    }


def format_article_report(index, article, score_result):
    score = "missing" if score_result is None else score_result.score
    score_band = "missing" if score_result is None else score_result.score_band
    report_band = "band_C" if score_band in {"band_C", "band_D", "drop", "missing"} else score_band
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
    lines.extend(format_policy_lines(score_result))
    lines.extend(
        [
            "",
            "DESCRIPTION:",
            str(article.description or "None").strip() or "None",
            "",
            "FULL CONTENT FROM WORLD NEWS API:",
            str(article.content or "None").strip() or "None",
            "",
        ]
    )
    return "\n".join(lines)


def format_policy_lines(score_result):
    if score_result is None:
        return ["Policy article: missing", "Policy: missing"]

    is_policy_article = bool(getattr(score_result, "is_policy_article", False))
    lines = [f"Policy article: {is_policy_article}"]
    policy = getattr(score_result, "policy", None)
    if policy is None:
        lines.append("Policy: None")
        return lines

    def policy_value(name, default=None):
        if isinstance(policy, dict):
            return policy.get(name, default)
        return getattr(policy, name, default)

    try:
        status_confidence = float(policy_value("status_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        status_confidence = 0.0

    lines.extend(
        [
            f"Policy category: {policy_value('category', 'other_policy')}",
            f"Policy status: {policy_value('status', 'unclear')}",
            f"Policy status confidence: {status_confidence:.2f}",
            wrap_text("Policy status reason", policy_value("status_reason")),
            f"Policy actor: {policy_value('actor') or 'None'}",
            f"Policy target: {policy_value('target') or 'None'}",
            f"Policy effective date: {policy_value('effective_date') or 'None'}",
        ]
    )
    return lines


def format_fast_filtered_report(
    index,
    article,
    decision,
    threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    lines = [
        "-" * 120,
        (
            f"#{index:03d} | fast_filter=filtered_out | "
            f"max_similarity={decision.max_similarity:.4f} | "
            f"threshold={threshold:.2f}"
        ),
        f"Matched negative anchor: {decision.matched_negative_text or 'None'}",
        f"Source: {article.source_name or 'Unknown'}",
        f"Published: {article.published_at or 'Unknown'}",
        f"URL: {article.url or 'None'}",
        wrap_text("Title", article.title),
        wrap_text("Fast filter text", article.fast_filter_text()),
        "",
        "DESCRIPTION:",
        str(article.description or "None").strip() or "None",
        "",
        "FULL CONTENT FROM WORLD NEWS API:",
        str(article.content or "None").strip() or "None",
        "",
    ]
    return "\n".join(lines)


def print_article(index, article, score_result):
    print(format_article_report(index, article, score_result))


def print_band_report(
    grouped,
    fast_filtered_items=None,
    requested_score_band=None,
    threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    fast_filtered_items = fast_filtered_items or []
    total = sum(len(items) for items in grouped.values())
    print("=" * 120)
    print("WORLD NEWS API FAST FILTER + PROMPT A SCORE DEBUG REPORT")
    print("Read-only check: World News API fetch + fast filter + Prompt A scoring only. No ingestion is performed.")
    print("Fast filter embeds title + first sentence of description. Prompt A scores title + description only.")
    print("band_C is the report label for program score_band='drop'.")
    print(f"Fast-filtered articles: {len(fast_filtered_items)}")
    print(f"Prompt A scored articles: {total}")
    print(
        "Counts: "
        + f"{FAST_FILTER_REPORT}={len(fast_filtered_items)} | "
        + " | ".join(f"{band}={len(grouped[band])}" for band in BAND_ORDER)
    )
    print("=" * 120)
    print()

    if requested_score_band is None:
        print()
        print("#" * 120)
        print(f"{FAST_FILTER_REPORT} ({len(fast_filtered_items)} articles)")
        print("#" * 120)
        print()
        if not fast_filtered_items:
            print("No articles filtered out by the fast filter.")
        else:
            for index, article, decision in fast_filtered_items:
                print(format_fast_filtered_report(index, article, decision, threshold=threshold))

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


def build_band_file_text(report_band, grouped, generated_at, fast_filtered_count=0):
    total = sum(len(items) for items in grouped.values())
    lines = [
        "WORLD NEWS API FAST FILTER + PROMPT A SCORE DEBUG REPORT",
        f"Generated at: {generated_at}",
        "Read-only check: World News API fetch + fast filter + Prompt A scoring only. No ingestion was performed.",
        "Fast filter embeds title + first sentence of description. Prompt A scores title + description only.",
        "band_C is the report label for program score_band='drop'.",
        f"Current file: {report_band}",
        f"Articles in this file: {len(grouped[report_band])}",
        f"Fast-filtered articles: {fast_filtered_count}",
        f"Prompt A scored articles: {total}",
        "Counts: "
        + f"{FAST_FILTER_REPORT}={fast_filtered_count} | "
        + " | ".join(f"{band}={len(grouped[band])}" for band in BAND_ORDER),
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


def build_fast_filter_file_text(
    fast_filtered_items,
    grouped,
    generated_at,
    threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    total_scored = sum(len(items) for items in grouped.values())
    lines = [
        "WORLD NEWS API FAST FILTER DEBUG REPORT",
        f"Generated at: {generated_at}",
        "Read-only check: World News API fetch + fast filter + Prompt A scoring only. No ingestion was performed.",
        "Fast filter embeds title + first sentence of description. Prompt A scores title + description only.",
        f"Current file: {FAST_FILTER_REPORT}",
        f"Fast-filter threshold: {threshold:.2f}",
        f"Articles in this file: {len(fast_filtered_items)}",
        f"Prompt A scored articles after fast filter: {total_scored}",
        "Counts: "
        + f"{FAST_FILTER_REPORT}={len(fast_filtered_items)} | "
        + " | ".join(f"{band}={len(grouped[band])}" for band in BAND_ORDER),
        "=" * 120,
        "",
    ]

    if not fast_filtered_items:
        lines.append("No articles filtered out by the fast filter.")
    else:
        for index, article, decision in fast_filtered_items:
            lines.append(format_fast_filtered_report(index, article, decision, threshold=threshold))

    lines.append("")
    return "\n".join(lines)


def write_band_report_files(
    grouped,
    output_dir=DEFAULT_OUTPUT_DIR,
    fast_filtered_items=None,
    threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    fast_filtered_items = fast_filtered_items or []
    os.makedirs(output_dir, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    written_paths = {}

    fast_filter_path = os.path.join(output_dir, BAND_OUTPUT_FILES[FAST_FILTER_REPORT])
    fast_filter_text = build_fast_filter_file_text(
        fast_filtered_items,
        grouped,
        generated_at,
        threshold=threshold,
    )
    with open(fast_filter_path, "w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(fast_filter_text)
    written_paths[FAST_FILTER_REPORT] = fast_filter_path
    log_step(f"Wrote {len(fast_filtered_items)} fast-filtered articles to {fast_filter_path}")

    for report_band in BAND_ORDER:
        path = os.path.join(output_dir, BAND_OUTPUT_FILES[report_band])
        file_text = build_band_file_text(
            report_band,
            grouped,
            generated_at,
            fast_filtered_count=len(fast_filtered_items),
        )
        with open(path, "w", encoding="utf-8", newline="\n") as output_file:
            output_file.write(file_text)
        written_paths[report_band] = path
        log_step(f"Wrote {len(grouped[report_band])} articles to {path}")

    return written_paths


def generate_band_report_files(
    limit=100,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
    output_dir=DEFAULT_OUTPUT_DIR,
    prompt_a_model=DEFAULT_PROMPT_A_MODEL,
    prompt_a_batch_size=DEFAULT_PROMPT_A_BATCH_SIZE,
    prompt_a_timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
    prompt_a_retries=DEFAULT_PROMPT_A_RETRIES,
    fast_filter_embed_model=DEFAULT_FAST_FILTER_EMBED_MODEL,
    negative_similarity_threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    pipeline = run_read_only_scoring_pipeline(
        limit=limit,
        search_query=search_query,
        prompt_a_model=prompt_a_model,
        prompt_a_batch_size=prompt_a_batch_size,
        prompt_a_timeout_seconds=prompt_a_timeout_seconds,
        prompt_a_retries=prompt_a_retries,
        fast_filter_embed_model=fast_filter_embed_model,
        negative_similarity_threshold=negative_similarity_threshold,
    )
    grouped = pipeline["grouped"]
    fast_filtered_items = pipeline["fast_filtered_items"]
    print_distribution_summary(grouped, fast_filtered_items)
    log_step(f"Writing band files into {output_dir}...")
    written_paths = write_band_report_files(
        grouped,
        output_dir=output_dir,
        fast_filtered_items=fast_filtered_items,
        threshold=negative_similarity_threshold,
    )
    log_step("Done.")
    return written_paths


def score_recent_news(
    limit=100,
    requested_band=None,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
    prompt_a_model=DEFAULT_PROMPT_A_MODEL,
    prompt_a_batch_size=DEFAULT_PROMPT_A_BATCH_SIZE,
    prompt_a_timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
    prompt_a_retries=DEFAULT_PROMPT_A_RETRIES,
    fast_filter_embed_model=DEFAULT_FAST_FILTER_EMBED_MODEL,
    negative_similarity_threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
):
    pipeline = run_read_only_scoring_pipeline(
        limit=limit,
        search_query=search_query,
        prompt_a_model=prompt_a_model,
        prompt_a_batch_size=prompt_a_batch_size,
        prompt_a_timeout_seconds=prompt_a_timeout_seconds,
        prompt_a_retries=prompt_a_retries,
        fast_filter_embed_model=fast_filter_embed_model,
        negative_similarity_threshold=negative_similarity_threshold,
    )
    grouped = pipeline["grouped"]
    fast_filtered_items = pipeline["fast_filtered_items"]
    log_step("Printing readable band report...")
    print_band_report(
        grouped,
        fast_filtered_items=fast_filtered_items,
        requested_score_band=requested_band,
        threshold=negative_similarity_threshold,
    )
    log_step("Done.")


def build_prompt_b_payload(article, score_result):
    policy = getattr(score_result, "policy", None)
    if hasattr(policy, "model_dump"):
        policy = policy.model_dump(mode="json")

    payload = article.scoring_payload()
    payload.update(
        {
            "score": score_result.score,
            "score_band": score_result.score_band,
            "market_relevance_reason": score_result.market_relevance_reason,
            "is_policy_article": getattr(score_result, "is_policy_article", False),
            "policy": policy,
        }
    )
    return payload


def run_prompt_b_extraction_batches(
    scored_article_payloads,
    llm_client,
    batch_size=DEFAULT_PROMPT_B_BATCH_SIZE,
):
    extracted_by_id = {}
    batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(scored_article_payloads), batch_size):
        batch = scored_article_payloads[batch_start: batch_start + batch_size]
        extracted_by_id.update(ingest_news.run_prompt_b_extraction(batch, llm_client))
    return extracted_by_id


def format_band_a_metadata_report(index, article, score_result, extracted_article):
    graph_fact_count = 0
    if extracted_article is not None:
        graph_fact_count = len(extracted_article.metadata.graph_facts)

    lines = [
        "-" * 120,
        (
            f"#{index:03d} | score={score_result.score} | "
            f"program_band={score_result.score_band} | graph_facts={graph_fact_count}"
        ),
        f"Source: {article.source_name or 'Unknown'}",
        f"Published: {article.published_at or 'Unknown'}",
        f"URL: {article.url or 'None'}",
        wrap_text("Title", article.title),
        wrap_text("Market relevance reason", score_result.market_relevance_reason),
        "",
    ]

    if extracted_article is None:
        lines.extend(
            [
                "PROMPT B METADATA:",
                "Missing. Prompt B did not return an Article for this band_A item.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "PROMPT B METADATA JSON:",
            json.dumps(
                extracted_article.metadata.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "",
        ]
    )
    return "\n".join(lines)


def print_band_a_metadata_report(band_a_items, extracted_by_id):
    print("=" * 120)
    print("BAND_A PROMPT B METADATA EXTRACTION DEBUG REPORT")
    print("Read-only check: Prompt B metadata extraction only. No vector or graph ingestion is performed.")
    print(f"band_A articles: {len(band_a_items)}")
    print(f"Prompt B extracted articles: {len(extracted_by_id)}")
    print("=" * 120)
    print()

    if not band_a_items:
        print("No band_A articles to extract metadata from.")
        return

    for index, article, score_result in band_a_items:
        extracted_article = extracted_by_id.get(article.article_id)
        print(format_band_a_metadata_report(index, article, score_result, extracted_article))


def print_band_a_extracted_metadata(
    limit=100,
    search_query=DEFAULT_RECENT_NEWS_QUERY,
    prompt_a_model=DEFAULT_PROMPT_A_MODEL,
    prompt_a_batch_size=DEFAULT_PROMPT_A_BATCH_SIZE,
    prompt_a_timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
    prompt_a_retries=DEFAULT_PROMPT_A_RETRIES,
    fast_filter_embed_model=DEFAULT_FAST_FILTER_EMBED_MODEL,
    negative_similarity_threshold=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
    prompt_b_model=None,
    prompt_b_batch_size=DEFAULT_PROMPT_B_BATCH_SIZE,
    prompt_b_timeout_seconds=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
    prompt_b_retries=DEFAULT_PROMPT_A_RETRIES,
):
    pipeline = run_read_only_scoring_pipeline(
        limit=limit,
        search_query=search_query,
        prompt_a_model=prompt_a_model,
        prompt_a_batch_size=prompt_a_batch_size,
        prompt_a_timeout_seconds=prompt_a_timeout_seconds,
        prompt_a_retries=prompt_a_retries,
        fast_filter_embed_model=fast_filter_embed_model,
        negative_similarity_threshold=negative_similarity_threshold,
    )
    grouped = pipeline["grouped"]
    fast_filtered_items = pipeline["fast_filtered_items"]
    print_distribution_summary(grouped, fast_filtered_items)

    band_a_items = grouped["band_A"]
    if not band_a_items:
        print_band_a_metadata_report(band_a_items, {})
        log_step("Done.")
        return {}

    scored_payloads = [
        build_prompt_b_payload(article, score_result)
        for _, article, score_result in band_a_items
    ]
    prompt_b_model = prompt_b_model or prompt_a_model
    prompt_b_batch_size = max(1, int(prompt_b_batch_size))
    total_batches = (len(scored_payloads) + prompt_b_batch_size - 1) // prompt_b_batch_size
    log_step(
        f"Running Prompt B metadata extraction for {len(scored_payloads)} band_A article(s) "
        f"in {total_batches} batch(es) of up to {prompt_b_batch_size}..."
    )
    llm_client = ProgressLLMClient(
        LlamaIndexLLMClient(
            prompt_b_model,
            timeout_seconds=prompt_b_timeout_seconds,
            max_retries=prompt_b_retries,
        ),
        total_batches=total_batches,
        task_name="Prompt B",
        action="Extracting metadata",
    )
    extracted_by_id = run_prompt_b_extraction_batches(
        scored_payloads,
        llm_client,
        batch_size=prompt_b_batch_size,
    )
    log_step(f"Prompt B returned extracted metadata for {len(extracted_by_id)} article(s).")
    print_band_a_metadata_report(band_a_items, extracted_by_id)
    log_step("Done.")
    return extracted_by_id


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch recent World News API articles, run the fast negative-anchor filter, "
            "run Prompt A scoring on passed articles, and write readable report files "
            "without ingestion."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of recent World News API articles to score. Max 100.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_RECENT_NEWS_QUERY,
        help="Optional World News API text query. Leave unset for the most recent source-filtered news.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where fast_filtered, band_A, band_B, band_C, and band_f files will be written.",
    )
    parser.add_argument(
        "--prompt-a-model",
        default=DEFAULT_PROMPT_A_MODEL,
        help=(
            "Model used for Prompt A scoring. Default: gpt-4.1-mini. "
            "Gemini model IDs like gemini-2.5-flash are also supported."
        ),
    )
    parser.add_argument(
        "--prompt-a-batch-size",
        type=int,
        default=DEFAULT_PROMPT_A_BATCH_SIZE,
        help="Articles per Prompt A API call. Default: 5.",
    )
    parser.add_argument(
        "--prompt-a-timeout",
        type=int,
        default=DEFAULT_PROMPT_A_TIMEOUT_SECONDS,
        help="OpenAI Prompt A request timeout in seconds. Default: 300.",
    )
    parser.add_argument(
        "--prompt-a-retries",
        type=int,
        default=DEFAULT_PROMPT_A_RETRIES,
        help="OpenAI Prompt A retry attempts for timeout/rate-limit/server errors. Default: 3.",
    )
    parser.add_argument(
        "--fast-filter-embed-model",
        default=DEFAULT_FAST_FILTER_EMBED_MODEL,
        help="SentenceTransformer embedding model used by the fast negative-anchor filter.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=DEFAULT_NEGATIVE_SIMILARITY_THRESHOLD,
        help="Max cosine similarity threshold for fast filtering. Default: 0.85.",
    )
    parser.add_argument(
        "--band-a-metadata",
        action="store_true",
        help=(
            "Run the same read-only scoring flow, then run Prompt B only for band_A "
            "articles and print extracted metadata instead of writing band files."
        ),
    )
    parser.add_argument(
        "--prompt-b-model",
        default=None,
        help="Model used for Prompt B metadata extraction. Defaults to --prompt-a-model.",
    )
    parser.add_argument(
        "--prompt-b-batch-size",
        type=int,
        default=DEFAULT_PROMPT_B_BATCH_SIZE,
        help="Articles per Prompt B metadata extraction call. Default: 3.",
    )
    args = parser.parse_args()

    if args.band_a_metadata:
        print_band_a_extracted_metadata(
            limit=args.limit,
            search_query=args.query,
            prompt_a_model=args.prompt_a_model,
            prompt_a_batch_size=args.prompt_a_batch_size,
            prompt_a_timeout_seconds=args.prompt_a_timeout,
            prompt_a_retries=args.prompt_a_retries,
            fast_filter_embed_model=args.fast_filter_embed_model,
            negative_similarity_threshold=args.negative_threshold,
            prompt_b_model=args.prompt_b_model,
            prompt_b_batch_size=args.prompt_b_batch_size,
            prompt_b_timeout_seconds=args.prompt_a_timeout,
            prompt_b_retries=args.prompt_a_retries,
        )
        return

    written_paths = generate_band_report_files(
        limit=args.limit,
        search_query=args.query,
        output_dir=args.output_dir,
        prompt_a_model=args.prompt_a_model,
        prompt_a_batch_size=args.prompt_a_batch_size,
        prompt_a_timeout_seconds=args.prompt_a_timeout,
        prompt_a_retries=args.prompt_a_retries,
        fast_filter_embed_model=args.fast_filter_embed_model,
        negative_similarity_threshold=args.negative_threshold,
    )
    print()
    print("Generated files:")
    for report_name in REPORT_ORDER:
        print(f"{report_name}: {written_paths[report_name]}")


if __name__ == "__main__":
    main()
