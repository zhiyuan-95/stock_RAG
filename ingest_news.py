import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import ingest_graph
import ingest_stock


load_dotenv("config.env")

DEFAULT_NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
DEFAULT_NEWS_DATA_DIR = os.getenv("NEWS_DATA_DIR", "./data_store/news")
DEFAULT_NEWS_VECTOR_DIR = os.getenv("NEWS_STORAGE_DIR", "./storage/news/general")
DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")
NEWS_STATE_FILENAME = "news_state.json"
RAW_ARTICLE_DIRNAME = "raw_articles"
NEWS_HISTORY_YEARS = 2
NEWS_BATCH_SIZE = 10
SUMMARY_WORD_LIMIT = 150
RETENTION_MONTHS = 24
ALLOWED_SOURCE_DOMAINS = [
    "bloomberg.com",
    "reuters.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "cnbc.com",
    "apnews.com",
    "bbc.com",
    "marketwatch.com",
    "fortune.com",
]
ALLOWED_SOURCE_NAMES = {
    "bloomberg",
    "reuters",
    "financial times",
    "the wall street journal",
    "wall street journal",
    "the economist",
    "economist",
    "cnbc",
    "associated press",
    "ap",
    "ap news",
    "bbc news",
    "bbc",
    "marketwatch",
    "fortune",
}
BUCKET_QUERIES = {
    "us": (
        '(economy OR tariffs OR trade OR congress OR white house OR federal reserve '
        'OR labor market OR regulation) AND ("United States" OR US OR America)'
    ),
    "international": (
        '(global trade OR china OR europe OR japan OR india OR geopolitics OR sanctions '
        'OR central bank OR emerging markets)'
    ),
    "business": (
        '(earnings OR merger OR acquisition OR bankruptcy OR layoffs OR supply chain '
        'OR antitrust OR capital spending OR pricing)'
    ),
    "technology": (
        '(AI OR semiconductor OR cloud OR cybersecurity OR software OR smartphone '
        'OR data center OR chip)'
    ),
}
FIXED_KEYWORDS = {
    "sectors": {
        "Technology": ["technology", "software", "semiconductor", "cloud", "ai", "artificial intelligence"],
        "Healthcare": ["healthcare", "biotech", "pharma", "medical device", "hospital"],
        "Financials": ["bank", "banking", "insurance", "asset manager", "brokerage", "financials"],
        "Energy": ["oil", "gas", "energy", "refining", "exploration", "opec"],
        "Consumer Discretionary": ["retail", "e-commerce", "consumer discretionary", "apparel", "automotive"],
        "Industrials": ["industrial", "aerospace", "defense", "rail", "machinery", "logistics"],
        "Materials": ["materials", "chemicals", "mining", "steel", "copper"],
        "Utilities": ["utility", "utilities", "power grid", "electricity"],
        "Real Estate": ["reit", "real estate", "commercial property", "residential property"],
        "Communication Services": ["media", "telecom", "streaming", "advertising platform", "communication services"],
        "Consumer Staples": ["consumer staples", "food", "beverage", "household products", "grocery"],
    },
    "event_types": {
        "Earnings": ["earnings", "earnings beat", "earnings miss", "guidance"],
        "Merger": ["merger", "merge"],
        "Acquisition": ["acquisition", "acquire", "buyout", "takeover"],
        "IPO": ["ipo", "initial public offering"],
        "Dividend": ["dividend", "payout"],
        "Product Launch": ["product launch", "launch", "rollout"],
        "Executive Change": ["ceo", "cfo", "executive", "leadership change", "board appointment"],
        "Layoff": ["layoff", "job cuts", "workforce reduction"],
        "Lawsuit": ["lawsuit", "legal action", "litigation", "settlement", "ruling"],
        "Bankruptcy": ["bankruptcy", "chapter 11", "restructuring"],
        "IP Dispute": ["patent", "copyright", "ip dispute", "trademark"],
    },
    "regulatory_bodies": {
        "SEC": ["sec", "securities and exchange commission"],
        "FDA": ["fda", "food and drug administration"],
        "FTC": ["ftc", "federal trade commission"],
        "EPA": ["epa", "environmental protection agency"],
        "DOJ": ["doj", "department of justice"],
        "EU Commission": ["eu commission", "european commission"],
        "FASB": ["fasb"],
        "IFRS": ["ifrs"],
        "Federal Reserve": ["federal reserve", "fed"],
        "ECB": ["ecb", "european central bank"],
        "BOJ": ["bank of japan", "boj"],
    },
    "indices_and_exchanges": {
        "S&P 500": ["s&p 500", "sp500"],
        "Dow Jones": ["dow jones", "dow"],
        "NASDAQ": ["nasdaq"],
        "NYSE": ["nyse"],
        "Russell 2000": ["russell 2000"],
        "FTSE 100": ["ftse 100"],
        "Nikkei": ["nikkei"],
    },
    "geographies_markets": {
        "United States": ["united states", "u.s.", "u.s", "us ", " america "],
        "China": ["china", "chinese"],
        "Europe": ["europe", "eurozone", "eu"],
        "Asia": ["asia", "asian"],
        "Emerging Markets": ["emerging markets"],
        "United Kingdom": ["united kingdom", "uk", "britain"],
        "Japan": ["japan", "japanese"],
        "India": ["india", "indian"],
        "Germany": ["germany", "german"],
        "Brazil": ["brazil", "brazilian"],
    },
    "commodities_benchmarks": {
        "WTI": ["wti"],
        "Brent": ["brent"],
        "Gold": ["gold"],
        "Silver": ["silver"],
        "Copper": ["copper"],
        "Crude Oil": ["crude oil"],
        "Natural Gas": ["natural gas", "lng"],
        "Corn": ["corn"],
        "Soybeans": ["soybeans", "soybean"],
    },
    "currencies_fx": {
        "USD": ["usd", "u.s. dollar", "us dollar"],
        "EUR": ["eur", "euro"],
        "JPY": ["jpy", "yen"],
        "GBP": ["gbp", "sterling", "pound"],
        "CNY": ["cny", "yuan", "renminbi"],
        "INR": ["inr", "rupee"],
        "CAD": ["cad", "canadian dollar"],
        "AUD": ["aud", "australian dollar"],
        "EUR/USD": ["eur/usd"],
        "USD/JPY": ["usd/jpy"],
    },
    "macro_indicators": {
        "GDP": ["gdp", "gross domestic product"],
        "CPI": ["cpi", "consumer price index", "inflation"],
        "PPI": ["ppi", "producer price index"],
        "Unemployment Rate": ["unemployment", "jobless rate"],
        "Nonfarm Payrolls": ["nonfarm payrolls", "nfp"],
        "Retail Sales": ["retail sales"],
        "ISM Manufacturing": ["ism manufacturing", "pmi"],
        "Fed Funds Rate": ["fed funds rate", "interest rate"],
        "PCE": ["pce", "personal consumption expenditures"],
    },
    "esg_frameworks": {
        "GRI": ["gri"],
        "SASB": ["sasb"],
        "TCFD": ["tcfd"],
        "ISSB": ["issb"],
        "UN SDG": ["un sdg", "sustainable development goals"],
        "Scope 1/2/3": ["scope 1", "scope 2", "scope 3"],
        "GHG Protocol": ["ghg protocol"],
        "CSRD": ["csrd"],
        "ESG": ["esg", "environmental social governance"],
        "Sustainability": ["sustainability", "sustainable"],
    },
    "industries": {
        "Semiconductors": ["semiconductor", "chipmaker", "chips"],
        "Software": ["software", "saas", "enterprise software"],
        "Banks": ["bank", "banking", "lender"],
        "Insurance": ["insurance", "insurer"],
        "Oil & Gas": ["oil", "gas", "refining", "upstream", "downstream"],
        "Retail": ["retail", "retailer", "e-commerce", "store chain"],
        "Automobiles": ["automobile", "auto maker", "ev maker", "carmaker"],
        "Airlines": ["airline", "carrier", "aviation"],
        "Telecommunications": ["telecom", "wireless", "broadband", "mobile network"],
        "Media & Entertainment": ["media", "streaming", "entertainment", "broadcast"],
    },
}
GENERAL_SCORE_PROMPT = """
You are a senior equity analyst.
Score each of the following news articles from 0-10 for long-term macro/sector fundamental importance.

{articles_block}

Reply in this exact format only:
Score 1: 7
Score 2: 4
Score 3: 9
"""
NEWS_SUMMARY_PROMPT = """
Summarize in no more than 150 words for stock analysts.
Be factual, concise, and focus on long-term implications for earnings, margins, supply chains,
competition, regulation, and sector structure.

Article:
{article_text}
"""
NEWS_FACT_PROMPT = """
Extract 3-5 bullet points of LONG-TERM FUNDAMENTAL impact only.
Focus on:
- earnings, margins, cash flow
- customers, competitors, suppliers
- technology moats or supply chain
- regulation, policy, or macro risk

Reply with bullet points only.

Article:
{article_text}
"""
NEWS_METADATA_PROMPT = """
You are extracting metadata for a long-term financial news knowledge base.
Return valid JSON only with these keys:
- dynamic_keywords: list[str]
- companies: list[str]
- customers: list[str]
- people: list[str]
- products: list[str]
- services: list[str]
- brands: list[str]
- emerging_topics: list[str]
- technology_topics: list[str]
- esg_issues: list[str]
- financial_phrases: list[str]
- competitor_mentions: list[{{"name": str, "relation": "positive"|"negative"|"neutral"}}]
- custom_competitive_mentions: list[str]
- monetary_policy_details: list[str]
- fiscal_policy_details: list[str]
- commodity_context: list[str]
- primary_ticker_focus: str
- sentiment_score: number
- sentiment_explanation: str
- financial_impact_keywords: list[str]
- relevance_to_financial_indicators: "high"|"medium"|"low"
- normalized_event_timeline: str

News title: {title}
News source: {source}
News summary: {summary}
News text:
{article_text}
"""


def _require_newsapi_key():
    api_key = str(DEFAULT_NEWSAPI_KEY or "").strip()
    if not api_key:
        raise ValueError("NEWSAPI_KEY is not set in config.env or the environment.")
    return api_key


def _news_state_path(data_dir=DEFAULT_NEWS_DATA_DIR):
    return os.path.join(data_dir, NEWS_STATE_FILENAME)


def _raw_article_dir(data_dir=DEFAULT_NEWS_DATA_DIR):
    return os.path.join(data_dir, RAW_ARTICLE_DIRNAME)


def _ensure_news_dirs(data_dir=DEFAULT_NEWS_DATA_DIR, vector_dir=DEFAULT_NEWS_VECTOR_DIR):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(_raw_article_dir(data_dir=data_dir), exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)


def _default_news_state():
    return {
        "last_daily_maintenance": None,
        "last_friday_maintenance": None,
        "last_initialized_at": None,
        "articles": {},
    }


def _load_news_state(data_dir=DEFAULT_NEWS_DATA_DIR):
    path = _news_state_path(data_dir=data_dir)
    if not os.path.isfile(path):
        return _default_news_state()
    with open(path, "r", encoding="utf-8") as state_file:
        loaded = json.load(state_file)
    state = _default_news_state()
    state.update({
        "last_daily_maintenance": loaded.get("last_daily_maintenance"),
        "last_friday_maintenance": loaded.get("last_friday_maintenance"),
        "last_initialized_at": loaded.get("last_initialized_at"),
        "articles": loaded.get("articles", {}),
    })
    return state


def _save_news_state(state, data_dir=DEFAULT_NEWS_DATA_DIR):
    _ensure_news_dirs(data_dir=data_dir)
    path = _news_state_path(data_dir=data_dir)
    with open(path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2, sort_keys=True)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _slugify(value):
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return normalized or "news"


def _parse_iso_datetime(value):
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_iso_datetime(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _normalize_source_name(name):
    normalized = _normalize_whitespace(name).lower()
    normalized = normalized.replace("&", "and")
    return normalized


def _source_domain(url):
    host = (urlparse(str(url or "")).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _source_is_allowed(article):
    source_name = _normalize_source_name(article.get("source_name"))
    source_domain = _source_domain(article.get("url"))
    if source_name in ALLOWED_SOURCE_NAMES:
        return True
    return any(source_domain.endswith(domain) for domain in ALLOWED_SOURCE_DOMAINS)


def _article_text(article):
    text_parts = [
        article.get("title"),
        article.get("description"),
        article.get("content"),
    ]
    return "\n\n".join(
        part.strip() for part in text_parts if str(part or "").strip()
    ).strip()


def _article_id(article):
    unique_value = article.get("url") or (
        f"{article.get('source_name')}|{article.get('title')}|{article.get('published_at')}"
    )
    digest = hashlib.sha1(str(unique_value).encode("utf-8")).hexdigest()
    return f"news::{digest}"


def _article_brief(article, max_chars=900):
    text = _normalize_whitespace(_article_text(article))
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return (
        f"Title: {article.get('title') or 'Unknown'}\n"
        f"Source: {article.get('source_name') or 'Unknown'}\n"
        f"Published: {article.get('published_at') or 'Unknown'}\n"
        f"Bucket: {article.get('bucket') or 'general'}\n"
        f"Text: {text}"
    )


def _json_from_response(text):
    raw_text = str(text or "").strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fenced_match:
        raw_text = fenced_match.group(1)
    else:
        object_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if object_match:
            raw_text = object_match.group(1)
    return json.loads(raw_text)


def _llm_complete(prompt):
    response = Settings.llm.complete(prompt)
    return str(response)


def _enforce_word_limit(text, max_words=SUMMARY_WORD_LIMIT):
    words = re.findall(r"\S+", str(text or "").strip())
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


def _score_articles_batch(articles):
    article_blocks = []
    for idx, article in enumerate(articles, start=1):
        article_blocks.append(f"Article {idx}:\n{_article_brief(article)}")
    prompt = GENERAL_SCORE_PROMPT.format(articles_block="\n\n".join(article_blocks))
    response_text = _llm_complete(prompt)
    score_matches = re.findall(r"Score\s+(\d+):\s*(10|[0-9])", response_text)
    scores = {}
    for idx_text, score_text in score_matches:
        idx = int(idx_text) - 1
        if 0 <= idx < len(articles):
            scores[articles[idx]["article_id"]] = int(score_text)
    for article in articles:
        scores.setdefault(article["article_id"], 0)
    return scores


def score_articles(articles, batch_size=NEWS_BATCH_SIZE):
    scores = {}
    for batch_start in range(0, len(articles), batch_size):
        batch = articles[batch_start: batch_start + batch_size]
        scores.update(_score_articles_batch(batch))
    return scores


def summarize_article(article_text):
    summary_prompt = NEWS_SUMMARY_PROMPT.format(article_text=article_text[:12000])
    summary_text = _normalize_whitespace(_llm_complete(summary_prompt))
    return _enforce_word_limit(summary_text, max_words=SUMMARY_WORD_LIMIT)


def extract_news_facts(article_text):
    facts_prompt = NEWS_FACT_PROMPT.format(article_text=article_text[:12000])
    facts_text = _llm_complete(facts_prompt)
    bullet_lines = []
    for line in str(facts_text).splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if not cleaned.startswith(("-", "*", "•")):
            cleaned = f"- {cleaned.lstrip('-*• ').strip()}"
        bullet_lines.append(cleaned)
    return "\n".join(bullet_lines[:5]).strip()


def _extract_fixed_keyword_metadata(article, text, company_aliases):
    raw_text = str(text or "")
    haystack = f" {raw_text.lower()} "
    metadata = {}

    for category, keyword_map in FIXED_KEYWORDS.items():
        matches = []
        for label, aliases in keyword_map.items():
            for alias in aliases:
                alias_haystack = f" {str(alias).lower()} "
                if alias_haystack in haystack:
                    matches.append(label)
                    break
        metadata[category] = sorted(set(matches))

    matched_tickers = []
    matched_companies = []
    for ticker, alias_groups in company_aliases.items():
        ticker_aliases = alias_groups.get("ticker_aliases", [])
        name_aliases = alias_groups.get("name_aliases", [])

        matched = False
        for alias in ticker_aliases:
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", raw_text):
                matched_tickers.append(ticker)
                matched_companies.append(alias)
                matched = True
                break
        if matched:
            continue

        for alias in name_aliases:
            alias_text = f" {alias.lower()} "
            if alias_text in haystack:
                matched_tickers.append(ticker)
                matched_companies.append(alias)
                break
    metadata["tickers"] = sorted(set(matched_tickers))
    metadata["core_companies"] = sorted(set(matched_companies))
    metadata["source_domain"] = _source_domain(article.get("url"))
    return metadata


def _build_company_aliases():
    aliases = {}
    try:
        ticker_map = ingest_stock._load_sec_ticker_map()
    except Exception:
        ticker_map = {}

    for ticker, payload in ticker_map.items():
        ticker_aliases = set()
        if ticker and len(ticker) >= 3:
            ticker_aliases.add(ticker)
        name_aliases = set()
        title = _normalize_whitespace(payload.get("title"))
        if title and len(title) <= 60:
            if len(title) >= 4:
                name_aliases.add(title)
            simplified = re.sub(
                r"\b(inc|corp|corporation|ltd|plc|co|company|holdings)\b\.?",
                "",
                title,
                flags=re.IGNORECASE,
            )
            simplified = _normalize_whitespace(simplified)
            if simplified and len(simplified) >= 4:
                name_aliases.add(simplified)
        aliases[ticker] = {
            "ticker_aliases": sorted(ticker_aliases),
            "name_aliases": sorted(name_aliases),
        }
    return aliases


def _extract_dynamic_metadata(article, summary_text, article_text):
    prompt = NEWS_METADATA_PROMPT.format(
        title=article.get("title") or "",
        source=article.get("source_name") or "",
        summary=summary_text or "",
        article_text=article_text[:9000],
    )
    try:
        parsed = _json_from_response(_llm_complete(prompt))
    except Exception:
        parsed = {}

    return {
        "dynamic_keywords": parsed.get("dynamic_keywords", []),
        "companies": parsed.get("companies", []),
        "customers": parsed.get("customers", []),
        "people": parsed.get("people", []),
        "products": parsed.get("products", []),
        "services": parsed.get("services", []),
        "brands": parsed.get("brands", []),
        "emerging_topics": parsed.get("emerging_topics", []),
        "technology_topics": parsed.get("technology_topics", []),
        "esg_issues": parsed.get("esg_issues", []),
        "financial_phrases": parsed.get("financial_phrases", []),
        "competitor_mentions": parsed.get("competitor_mentions", []),
        "custom_competitive_mentions": parsed.get("custom_competitive_mentions", []),
        "monetary_policy_details": parsed.get("monetary_policy_details", []),
        "fiscal_policy_details": parsed.get("fiscal_policy_details", []),
        "commodity_context": parsed.get("commodity_context", []),
        "primary_ticker_focus": parsed.get("primary_ticker_focus"),
        "sentiment_score": parsed.get("sentiment_score"),
        "sentiment_explanation": parsed.get("sentiment_explanation"),
        "financial_impact_keywords": parsed.get("financial_impact_keywords", []),
        "relevance_to_financial_indicators": parsed.get("relevance_to_financial_indicators"),
        "normalized_event_timeline": parsed.get("normalized_event_timeline"),
    }


def extract_news_metadata(article, summary_text, article_text, company_aliases):
    fixed_metadata = _extract_fixed_keyword_metadata(article, f"{article_text}\n\n{summary_text}", company_aliases)
    dynamic_metadata = _extract_dynamic_metadata(article, summary_text, article_text)
    metadata = {
        "news_bucket": article.get("bucket"),
        "source_name": article.get("source_name"),
        "published_at": article.get("published_at"),
        "url": article.get("url"),
    }
    metadata.update(fixed_metadata)
    metadata.update(dynamic_metadata)
    return metadata


def _retention_mode(score, published_dt, now_dt):
    age_days = max((now_dt - published_dt).days, 0)
    within_retention = age_days <= RETENTION_MONTHS * 30
    if score < 5:
        return None, False
    if score >= 8:
        return ("full" if within_retention else "summary"), True
    if within_retention:
        return "summary", False
    return None, False


def _save_raw_article_record(record, data_dir=DEFAULT_NEWS_DATA_DIR):
    raw_dir = _raw_article_dir(data_dir=data_dir)
    os.makedirs(raw_dir, exist_ok=True)
    filename = f"{_slugify(record['published_at'])}-{record['article_id'].split('::')[-1]}.json"
    path = os.path.join(raw_dir, filename)
    with open(path, "w", encoding="utf-8") as raw_file:
        json.dump(record, raw_file, ensure_ascii=False, indent=2, sort_keys=True)


def _newsapi_request(params):
    response = requests.get(
        "https://newsapi.org/v2/everything",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI request failed: {payload}")
    return payload


def fetch_news_bucket(bucket, start_dt, end_dt, max_pages=5, page_size=100):
    api_key = _require_newsapi_key()
    query = BUCKET_QUERIES[bucket]
    articles = []
    for page in range(1, max_pages + 1):
        payload = _newsapi_request(
            {
                "apiKey": api_key,
                "q": query,
                "from": start_dt.date().isoformat(),
                "to": end_dt.date().isoformat(),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(page_size, 100),
                "page": page,
                "domains": ",".join(ALLOWED_SOURCE_DOMAINS),
            }
        )
        page_articles = payload.get("articles", [])
        if not page_articles:
            break

        for raw_article in page_articles:
            article = {
                "bucket": bucket,
                "title": _normalize_whitespace(raw_article.get("title")),
                "description": _normalize_whitespace(raw_article.get("description")),
                "content": _normalize_whitespace(raw_article.get("content")),
                "url": raw_article.get("url"),
                "source_name": _normalize_whitespace((raw_article.get("source") or {}).get("name")),
                "published_at": raw_article.get("publishedAt"),
            }
            article["article_id"] = _article_id(article)
            if not _source_is_allowed(article):
                continue
            if not _article_text(article):
                continue
            articles.append(article)

        if len(page_articles) < min(page_size, 100):
            break
    return articles


def fetch_broad_news(start_dt, end_dt, buckets=None, max_pages=5, page_size=100):
    collected = {}
    for bucket in (buckets or BUCKET_QUERIES.keys()):
        for article in fetch_news_bucket(
            bucket=bucket,
            start_dt=start_dt,
            end_dt=end_dt,
            max_pages=max_pages,
            page_size=page_size,
        ):
            existing = collected.get(article["article_id"])
            if existing is None:
                collected[article["article_id"]] = article
                continue
            existing_bucket = existing.get("bucket", "")
            if bucket not in existing_bucket.split(","):
                existing["bucket"] = ",".join(sorted(set(existing_bucket.split(",") + [bucket]) - {""}))
    return list(collected.values())


def _rebuild_news_vector_index(state, vector_dir=DEFAULT_NEWS_VECTOR_DIR):
    documents = []
    for record in state.get("articles", {}).values():
        vector_mode = record.get("vector_mode")
        if vector_mode not in {"full", "summary"}:
            continue
        text = record.get("full_text") if vector_mode == "full" else record.get("summary")
        if not str(text or "").strip():
            continue
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "article_id": record.get("article_id"),
                "type": "news_full_text" if vector_mode == "full" else "news_summary",
                "title": record.get("title"),
                "source": record.get("source_name"),
                "published_at": record.get("published_at"),
                "news_bucket": record.get("bucket"),
                "score": record.get("score"),
                "url": record.get("url"),
                "news_storage_mode": vector_mode,
            }
        )
        documents.append(Document(text=text, metadata=metadata))

    if not documents:
        if os.path.isdir(vector_dir):
            shutil.rmtree(vector_dir)
        return None

    ingest_stock.env()
    node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=70)
    nodes = node_parser.get_nodes_from_documents(documents)
    if os.path.isdir(vector_dir):
        shutil.rmtree(vector_dir)
    os.makedirs(vector_dir, exist_ok=True)
    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=vector_dir)
    return vector_dir


def _rebuild_news_graph(state, storage_dir=DEFAULT_GRAPH_STORAGE_DIR):
    documents = []
    for record in state.get("articles", {}).values():
        if not record.get("graph_keep"):
            continue
        facts_text = str(record.get("graph_facts") or "").strip()
        if not facts_text:
            continue
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "type": "news_fundamental_facts",
                "title": record.get("title"),
                "source": record.get("source_name"),
                "published_at": record.get("published_at"),
                "news_bucket": record.get("bucket"),
                "score": record.get("score"),
                "url": record.get("url"),
            }
        )
        primary_ticker = str(metadata.get("primary_ticker_focus") or "").upper().strip()
        if primary_ticker:
            metadata["ticker"] = primary_ticker
        documents.append(Document(text=facts_text, metadata=metadata))

    return ingest_graph.upsert_news_documents(
        scope="broad_news",
        documents=documents,
        storage_dir=storage_dir,
    )


def _process_articles(articles, state, now_dt, data_dir=DEFAULT_NEWS_DATA_DIR):
    if not articles:
        return 0

    company_aliases = _build_company_aliases()
    new_articles = [
        article for article in articles
        if article["article_id"] not in state.get("articles", {})
    ]
    if not new_articles:
        return 0

    scores = score_articles(new_articles)
    processed_count = 0
    for article in new_articles:
        score = int(scores.get(article["article_id"], 0))
        published_dt = _parse_iso_datetime(article.get("published_at")) or now_dt
        vector_mode, graph_keep = _retention_mode(score, published_dt, now_dt)
        if score < 5:
            state["articles"][article["article_id"]] = {
                "article_id": article["article_id"],
                "title": article.get("title"),
                "url": article.get("url"),
                "source_name": article.get("source_name"),
                "published_at": article.get("published_at"),
                "bucket": article.get("bucket"),
                "score": score,
                "vector_mode": None,
                "graph_keep": False,
                "metadata": {"news_bucket": article.get("bucket")},
            }
            _save_raw_article_record(state["articles"][article["article_id"]], data_dir=data_dir)
            processed_count += 1
            continue

        full_text = _article_text(article)
        summary = summarize_article(full_text)
        metadata = extract_news_metadata(article, summary, full_text, company_aliases)
        graph_facts = extract_news_facts(full_text) if graph_keep else None

        record = {
            "article_id": article["article_id"],
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "full_text": full_text,
            "summary": summary,
            "graph_facts": graph_facts,
            "url": article.get("url"),
            "source_name": article.get("source_name"),
            "published_at": article.get("published_at"),
            "bucket": article.get("bucket"),
            "score": score,
            "vector_mode": vector_mode,
            "graph_keep": graph_keep,
            "metadata": metadata,
            "ingested_at": _format_iso_datetime(now_dt),
        }
        state["articles"][article["article_id"]] = record
        _save_raw_article_record(record, data_dir=data_dir)
        processed_count += 1
    return processed_count


def _apply_retention(state, now_dt):
    for record in state.get("articles", {}).values():
        published_dt = _parse_iso_datetime(record.get("published_at")) or now_dt
        vector_mode, graph_keep = _retention_mode(int(record.get("score", 0)), published_dt, now_dt)
        record["vector_mode"] = vector_mode
        record["graph_keep"] = graph_keep
        if not graph_keep:
            record["graph_facts"] = None


def _daterange_windows(start_dt, end_dt, window_days=7):
    cursor = start_dt
    while cursor < end_dt:
        window_end = min(cursor + timedelta(days=window_days), end_dt)
        yield cursor, window_end
        cursor = window_end


def _last_friday_on_or_before(target_date):
    days_back = (target_date.weekday() - 4) % 7
    return target_date - timedelta(days=days_back)


def initialize_broad_news_history(
    years=NEWS_HISTORY_YEARS,
    data_dir=DEFAULT_NEWS_DATA_DIR,
    vector_dir=DEFAULT_NEWS_VECTOR_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    ingest_stock.env()
    _ensure_news_dirs(data_dir=data_dir, vector_dir=vector_dir)
    state = _load_news_state(data_dir=data_dir)
    now_dt = datetime.now(timezone.utc)
    start_dt = now_dt - timedelta(days=365 * years)

    for window_start, window_end in _daterange_windows(start_dt, now_dt, window_days=7):
        articles = fetch_broad_news(window_start, window_end)
        _process_articles(articles, state, now_dt=now_dt, data_dir=data_dir)

    _apply_retention(state, now_dt=now_dt)
    _rebuild_news_vector_index(state, vector_dir=vector_dir)
    _rebuild_news_graph(state, storage_dir=graph_storage_dir)
    state["last_initialized_at"] = _format_iso_datetime(now_dt)
    state["last_daily_maintenance"] = now_dt.date().isoformat()
    state["last_friday_maintenance"] = _last_friday_on_or_before(now_dt.date()).isoformat()
    _save_news_state(state, data_dir=data_dir)
    return state


def run_friday_maintenance(
    friday_date=None,
    data_dir=DEFAULT_NEWS_DATA_DIR,
    vector_dir=DEFAULT_NEWS_VECTOR_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    ingest_stock.env()
    _ensure_news_dirs(data_dir=data_dir, vector_dir=vector_dir)
    state = _load_news_state(data_dir=data_dir)
    run_date = friday_date or _last_friday_on_or_before(datetime.now(timezone.utc).date())
    start_dt = datetime.combine(run_date - timedelta(days=6), datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(run_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    articles = fetch_broad_news(start_dt, end_dt)
    _process_articles(articles, state, now_dt=datetime.now(timezone.utc), data_dir=data_dir)
    _apply_retention(state, now_dt=datetime.now(timezone.utc))
    _rebuild_news_vector_index(state, vector_dir=vector_dir)
    _rebuild_news_graph(state, storage_dir=graph_storage_dir)
    state["last_friday_maintenance"] = run_date.isoformat()
    _save_news_state(state, data_dir=data_dir)
    return state


def run_daily_maintenance(
    data_dir=DEFAULT_NEWS_DATA_DIR,
    vector_dir=DEFAULT_NEWS_VECTOR_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    ingest_stock.env()
    _ensure_news_dirs(data_dir=data_dir, vector_dir=vector_dir)
    state = _load_news_state(data_dir=data_dir)
    now_dt = datetime.now(timezone.utc)
    today = now_dt.date()

    if state.get("last_daily_maintenance") == today.isoformat():
        return state

    last_friday = _last_friday_on_or_before(today)
    if state.get("last_friday_maintenance") != last_friday.isoformat():
        run_friday_maintenance(
            friday_date=last_friday,
            data_dir=data_dir,
            vector_dir=vector_dir,
            graph_storage_dir=graph_storage_dir,
        )
        state = _load_news_state(data_dir=data_dir)

    last_daily_text = state.get("last_daily_maintenance")
    if last_daily_text:
        start_date = date.fromisoformat(last_daily_text)
    else:
        start_date = today - timedelta(days=1)

    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = now_dt
    articles = fetch_broad_news(start_dt, end_dt)
    _process_articles(articles, state, now_dt=now_dt, data_dir=data_dir)
    _apply_retention(state, now_dt=now_dt)
    _rebuild_news_vector_index(state, vector_dir=vector_dir)
    _rebuild_news_graph(state, storage_dir=graph_storage_dir)
    state["last_daily_maintenance"] = today.isoformat()
    _save_news_state(state, data_dir=data_dir)
    return state


def ingest_news(
    initialize=False,
    data_dir=DEFAULT_NEWS_DATA_DIR,
    vector_dir=DEFAULT_NEWS_VECTOR_DIR,
    graph_storage_dir=DEFAULT_GRAPH_STORAGE_DIR,
):
    if initialize:
        return initialize_broad_news_history(
            data_dir=data_dir,
            vector_dir=vector_dir,
            graph_storage_dir=graph_storage_dir,
        )
    return run_daily_maintenance(
        data_dir=data_dir,
        vector_dir=vector_dir,
        graph_storage_dir=graph_storage_dir,
    )


if __name__ == "__main__":
    ingest_news()
