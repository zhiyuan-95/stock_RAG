import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Sequence
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

try:
    from llama_index.core import Document, Settings, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:
    Document = None
    Settings = None
    VectorStoreIndex = None
    SentenceSplitter = None

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:
    import ingest_graph
except ImportError:
    ingest_graph = None

try:
    import ingest_stock
except ImportError:
    ingest_stock = None

from prompt import (
    GENERAL_SCORE_PROMPT,
    NEWS_FACT_PROMPT,
    NEWS_METADATA_PROMPT,
    NEWS_SUMMARY_PROMPT,
    PROMPT_A_TEMPLATE,
    PROMPT_B_TEMPLATE,
)


load_dotenv("config.env")


def _config_env_value(name, path="config.env"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if key.strip().upper() == name.upper():
                return value.strip().strip("\"'")
    return None


DEFAULT_WORLD_NEWS_API_KEY = (
    os.getenv("WORLD_NEWS_API_KEY")
    or os.getenv("World_News_API_KEY")
    or _config_env_value("WORLD_NEWS_API_KEY")
)
DEFAULT_NEWS_DATA_DIR = os.getenv("NEWS_DATA_DIR", "./data_store/news")
DEFAULT_NEWS_VECTOR_DIR = os.getenv("NEWS_STORAGE_DIR", "./storage/news/general")
DEFAULT_GRAPH_STORAGE_DIR = os.getenv("GRAPH_STORAGE_DIR", "./storage/graph")
NEWS_STATE_FILENAME = "news_state.json"
RAW_ARTICLE_DIRNAME = "raw_articles"
NEWS_HISTORY_YEARS = 2
NEWS_BATCH_SIZE = 10
DEFAULT_NEGATIVE_FILTER_EMBED_MODEL = "all-MiniLM-L6-v2"
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
WORLD_NEWS_SEARCH_URL = "https://api.worldnewsapi.com/search-news"
WORLD_NEWS_SOURCE_URLS = [
    "https://www.bloomberg.com",
    "https://www.reuters.com",
    "https://www.ft.com",
    "https://www.wsj.com",
    "https://www.economist.com",
    "https://apnews.com",
]
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


def _require_world_news_api_key():
    api_key = str(DEFAULT_WORLD_NEWS_API_KEY or "").strip()
    if not api_key:
        raise ValueError("WORLD_NEWS_API_KEY is not set in config.env or the environment.")
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


def _first_sentence(text):
    text = _normalize_whitespace(text)
    if not text:
        return ""
    sentence_boundary = re.search(r"(?<=[.!?])\s+", text)
    if sentence_boundary:
        return text[: sentence_boundary.start()].strip()
    return text


def _title_description_text(title, description):
    return "\n\n".join(
        text
        for text in [_normalize_whitespace(title), _normalize_whitespace(description)]
        if text
    ).strip()


def _title_first_description_sentence_text(title, description):
    return "\n\n".join(
        text
        for text in [_normalize_whitespace(title), _first_sentence(description)]
        if text
    ).strip()


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


def _source_name_from_url(url):
    domain = _source_domain(url)
    if domain.endswith("bloomberg.com"):
        return "Bloomberg"
    if domain.endswith("reuters.com"):
        return "Reuters"
    if domain.endswith("ft.com"):
        return "Financial Times"
    if domain.endswith("wsj.com"):
        return "The Wall Street Journal"
    if domain.endswith("economist.com"):
        return "The Economist"
    if domain.endswith("apnews.com"):
        return "Associated Press"
    return domain or None


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
    text = _normalize_whitespace(
        _title_description_text(article.get("title"), article.get("description"))
    )
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


SCORE_BAND_F = "band_f"
SCORE_BAND_DROP = "drop"
SCORE_BAND_B = "band_B"
SCORE_BAND_A = "band_A"
GRAPH_FACT_FIELDS = {
    "subject",
    "predicate",
    "object",
    "source_article_id",
    "published_at",
    "confidence",
}
POLICY_CATEGORIES = {
    "monetary_policy",
    "fiscal_policy_tax",
    "trade_tariffs_sanctions",
    "export_controls",
    "immigration_labor_policy",
    "energy_policy",
    "financial_regulation",
    "antitrust_regulation",
    "healthcare_regulation",
    "defense_geopolitical_policy",
    "industrial_policy",
    "other_policy",
}
POLICY_STATUSES = {
    "rhetoric",
    "threatened",
    "proposed",
    "announced",
    "signed",
    "implemented",
    "delayed",
    "reversed",
    "blocked",
    "expired",
    "unclear",
}
WORLD_NEWS_BUCKET_QUERIES = {
    "us": 'economy OR tariffs OR trade OR congress OR "Federal Reserve" OR regulation',
    "international": "global trade OR China OR Europe OR Japan OR India OR sanctions OR central bank",
    "business": "earnings OR merger OR acquisition OR bankruptcy OR layoffs OR supply chain",
    "technology": "AI OR semiconductor OR cloud OR cybersecurity OR software OR data center OR chip",
}
TYPO_KEY_NORMALIZATIONS = {
    "affacted_regions": "affected_regions",
    "affact_regions": "affected_regions",
    "artical_id": "article_id",
    "netative_anchor_vector": "negative_anchor_vector",
    "policy_status": "status",
    "policy_status_confidence": "status_confidence",
}
NULL_TEXT_VALUES = {"", "nan", "none", "null", "n/a", "na"}
ALLOWED_REGIONS = {
    "United States",
    "Canada",
    "Europe",
    "China",
    "Japan_Korea",
    "Asia_Pacific_ex_China_Japan_Korea",
    "Middle_East",
    "Latin_America",
    "Africa",
    "Global",
}
ALLOWED_MARKET_TRANSMISSION_CHANNELS = {
    "monetary_policy",
    "fiscal_policy",
    "inflation",
    "labor_market",
    "consumer_demand",
    "earnings_margin",
    "supply_chain",
    "commodities_energy",
    "credit_liquidity",
    "currency_fx",
    "regulation_policy",
    "trade_tariffs_sanctions",
    "technology_infrastructure",
    "market_sentiment",
    "other",
}
ALLOWED_IMPACTS = {"positive", "negative", "mixed", "neutral", "uncertain"}

NEGATIVE_TEXTS = [
    "Local high school basketball tournament",
    "Professional sports team game recap",
    "Athlete interview and trade rumors",
    "Golf tournament leaderboard and scores",
    "Fantasy football draft player rankings",
    "Celebrity red carpet fashion event",
    "Pop music album release gossip",
    "Movie release and theater reviews",
    "Television show episode recap commentary",
    "Reality TV star drama updates",
    "Local traffic accident and roadwork",
    "Regional weather forecast and alerts",
    "Local community crime report arrest",
    "Community fundraising event and charity",
    "School board meeting community minutes",
    "Personal wellness and dietary tips",
    "Daily horoscope and astrology reading",
    "Home gardening and landscaping advice",
    "Local restaurant opening and review",
    "Fitness routine and workout advice",
    "Consumer smartphone unboxing and review",
    "Video game console release announcement",
    "Esports tournament broadcast and results",
    "Mobile app update bugfix patch",
    "Partisan political campaign rally speech",
    "Local mayoral election debate coverage",
    "Politician personal scandal and gossip",
]


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class SentenceTransformerEmbeddingClient:
    def __init__(self, model: str = DEFAULT_NEGATIVE_FILTER_EMBED_MODEL, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the fast negative-anchor filter. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model = model
        self.batch_size = batch_size
        self.encoder = SentenceTransformer(self.model)

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        texts = [str(text or " ") for text in texts]
        if not texts:
            return []
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in embeddings]


class VectorStoreClient(Protocol):
    def upsert_article(
        self,
        article: "Article",
        text: str,
        retention_months: Optional[int],
        permanent: bool,
    ) -> None:
        ...


class GraphStoreClient(Protocol):
    def upsert_graph_facts(self, graph_facts: Sequence["GraphFact"]) -> None:
        ...


class QuarantineStoreClient(Protocol):
    def append(self, text: str) -> None:
        ...


def score_to_band(score: int) -> str:
    score = int(score)
    if 0 <= score <= 1:
        return SCORE_BAND_F
    if 2 <= score <= 4:
        return SCORE_BAND_DROP
    if 5 <= score <= 7:
        return SCORE_BAND_B
    if 8 <= score <= 10:
        return SCORE_BAND_A
    raise ValueError(f"score must be between 0 and 10, got {score}")


def _normalize_null_value(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, str) and value.strip().lower() in NULL_TEXT_VALUES:
        return None
    return value


def normalize_ingest_payload(value):
    value = _normalize_null_value(value)
    if isinstance(value, list):
        return [normalize_ingest_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {}
    for raw_key, raw_value in value.items():
        key = TYPO_KEY_NORMALIZATIONS.get(str(raw_key), str(raw_key))
        if key in {"event_id", "implied_ticker_candidates"}:
            continue
        normalized[key] = normalize_ingest_payload(raw_value)
    return normalized


def _stable_article_id(payload: Dict[str, Any]) -> str:
    unique_value = payload.get("url") or (
        f"{payload.get('source_name')}|{payload.get('title')}|{payload.get('published_at')}"
    )
    digest = hashlib.sha1(str(unique_value).encode("utf-8")).hexdigest()
    return f"news::{digest}"


def _clean_optional_text(value):
    value = _normalize_null_value(value)
    if value is None:
        return None
    cleaned = _normalize_whitespace(value)
    return cleaned or None


def _filter_allowed_strings(values, allowed_values):
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [value for value in values if value in allowed_values]


def _normalize_policy_signal_value(value):
    value = _normalize_null_value(value)
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, dict):
        return normalize_ingest_payload(value)
    if isinstance(value, str):
        cleaned = _clean_optional_text(value)
        if cleaned is None:
            return None
        return {
            "category": "other_policy",
            "status": "unclear",
            "status_confidence": 0.0,
            "status_reason": cleaned,
        }
    if isinstance(value, list):
        cleaned_items = [
            cleaned_item
            for item in value
            if (cleaned_item := _clean_optional_text(item)) is not None
        ]
        if cleaned_items:
            return {
                "category": "other_policy",
                "status": "unclear",
                "status_confidence": 0.0,
                "status_reason": "; ".join(cleaned_items),
            }
    return None


def _company_is_directly_mentioned(company: "CompanyMention", article_text: str) -> bool:
    haystack = f" {_normalize_whitespace(article_text).lower()} "
    company_name = _normalize_whitespace(company.name).lower()
    simplified_name = re.sub(
        r"\b(inc|inc\.|corp|corp\.|corporation|ltd|ltd\.|plc|co|co\.|company)\b",
        "",
        company_name,
    )
    simplified_name = _normalize_whitespace(simplified_name).lower()
    ticker = str(company.ticker or "").upper().strip()

    if company_name and f" {company_name} " in haystack:
        return True
    if simplified_name and f" {simplified_name} " in haystack:
        return True
    if ticker and re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", article_text):
        return True
    return False


class RawArticle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    article_id: Optional[str] = None
    published_at: Optional[str] = None
    source_name: Optional[str] = None
    title: str = ""
    url: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    bucket: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        data = normalize_ingest_payload(data or {})
        if not isinstance(data, dict):
            return data
        if "publishedAt" in data and "published_at" not in data:
            data["published_at"] = data["publishedAt"]
        source = data.get("source")
        if isinstance(source, dict) and "source_name" not in data:
            data["source_name"] = source.get("name")
        if not data.get("article_id"):
            data["article_id"] = _stable_article_id(data)
        return data

    @field_validator("published_at", "source_name", "url", "description", "content", "bucket", mode="before")
    @classmethod
    def clean_optional_text_fields(cls, value):
        return _clean_optional_text(value)

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, value):
        return _normalize_whitespace(value) or "Untitled"

    def vector_text(self) -> str:
        return "\n\n".join(
            text
            for text in [self.title, self.description, self.content]
            if str(text or "").strip()
        ).strip()

    def fast_filter_text(self) -> str:
        return _title_first_description_sentence_text(self.title, self.description)

    def prompt_a_payload(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "description": self.description,
            "source_name": self.source_name,
            "published_at": self.published_at,
            "url": self.url,
        }

    def scoring_payload(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "source_name": self.source_name,
            "published_at": self.published_at,
            "url": self.url,
        }


class PolicySignal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str = "other_policy"
    status: str = "unclear"
    status_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status_reason: Optional[str] = None
    actor: Optional[str] = None
    target: Optional[str] = None
    effective_date: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return _normalize_policy_signal_value(data) or {}

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value):
        value = _clean_optional_text(value)
        return value if value in POLICY_CATEGORIES else "other_policy"

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        value = _clean_optional_text(value)
        return value if value in POLICY_STATUSES else "unclear"

    @field_validator("status_reason", "actor", "target", "effective_date", mode="before")
    @classmethod
    def clean_optional_text_fields(cls, value):
        return _clean_optional_text(value)


class PromptAScoreResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    article_id: str
    score: int = Field(ge=0, le=10)
    score_band: str
    market_relevance_reason: Optional[str] = None
    extracted_negative_text: Optional[str] = None
    is_policy_article: bool = False
    policy: Optional[PolicySignal] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        data = normalize_ingest_payload(data or {})
        if isinstance(data, dict) and str(data.get("score_band", "")).lower() in {
            "band_c",
            "band_d",
            "c",
            "d",
        }:
            data["score_band"] = SCORE_BAND_DROP
        return data

    @field_validator("is_policy_article", mode="before")
    @classmethod
    def normalize_policy_flag(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @field_validator("policy", mode="before")
    @classmethod
    def normalize_policy(cls, value):
        return _normalize_policy_signal_value(value)

    @model_validator(mode="after")
    def normalize_band(self):
        self.score_band = score_to_band(self.score)
        if self.score_band == SCORE_BAND_F:
            self.market_relevance_reason = None
        else:
            self.extracted_negative_text = None
        if self.policy is not None and not self.is_policy_article:
            self.is_policy_article = True
        if not self.is_policy_article:
            self.policy = None
        return self


class PromptAResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: List[PromptAScoreResult] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})


class IndustryImpact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    industry: str
    sector: str
    impact: Literal["positive", "negative", "mixed", "neutral", "uncertain"] = "uncertain"
    reason: str

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})

    @field_validator("industry", "sector", "reason", mode="before")
    @classmethod
    def clean_required_text(cls, value):
        return _normalize_whitespace(value)

    @field_validator("impact", mode="before")
    @classmethod
    def normalize_impact(cls, value):
        value = _clean_optional_text(value)
        return value if value in ALLOWED_IMPACTS else "uncertain"


class CompanyMention(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    ticker: Optional[str] = None
    impact: Literal["positive", "negative", "mixed", "neutral", "uncertain"] = "uncertain"
    reason: str

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})

    @field_validator("name", "reason", mode="before")
    @classmethod
    def clean_required_text(cls, value):
        return _normalize_whitespace(value)

    @field_validator("ticker", mode="before")
    @classmethod
    def clean_ticker(cls, value):
        value = _clean_optional_text(value)
        return value.upper() if value else None

    @field_validator("impact", mode="before")
    @classmethod
    def normalize_impact(cls, value):
        value = _clean_optional_text(value)
        return value if value in ALLOWED_IMPACTS else "uncertain"


class GraphFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: str
    predicate: str
    object: str
    source_article_id: str
    published_at: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})

    @field_validator("subject", "predicate", "object", "source_article_id", mode="before")
    @classmethod
    def clean_required_text(cls, value):
        return _normalize_whitespace(value)

    @field_validator("published_at", mode="before")
    @classmethod
    def clean_published_at(cls, value):
        return _clean_optional_text(value)


class ArticleMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_policy_article: bool = False
    policy: Optional[PolicySignal] = None
    origin_regions: List[str] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    affected_industry_primary: Optional[IndustryImpact] = None
    affected_industry_secondary: List[IndustryImpact] = Field(default_factory=list)
    market_transmission_channel: List[str] = Field(default_factory=list)
    explicit_companies_mentioned: List[CompanyMention] = Field(default_factory=list)
    graph_facts: List[GraphFact] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})

    @field_validator("is_policy_article", mode="before")
    @classmethod
    def normalize_policy_flag(cls, value):
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @field_validator("policy", mode="before")
    @classmethod
    def normalize_policy(cls, value):
        return _normalize_policy_signal_value(value)

    @model_validator(mode="after")
    def enforce_policy_consistency(self):
        if self.policy is not None and not self.is_policy_article:
            self.is_policy_article = True
        if not self.is_policy_article:
            self.policy = None
        return self

    @field_validator("origin_regions", "affected_regions", mode="before")
    @classmethod
    def normalize_regions(cls, value):
        return _filter_allowed_strings(value, ALLOWED_REGIONS)

    @field_validator("market_transmission_channel", mode="before")
    @classmethod
    def normalize_channels(cls, value):
        return _filter_allowed_strings(value, ALLOWED_MARKET_TRANSMISSION_CHANNELS)

    @field_validator("graph_facts", mode="before")
    @classmethod
    def normalize_graph_facts(cls, value):
        facts = normalize_ingest_payload(value or [])
        if not isinstance(facts, list):
            return []
        return [
            {key: fact.get(key) for key in GRAPH_FACT_FIELDS if key in fact}
            for fact in facts
            if isinstance(fact, dict) and float(fact.get("confidence") or 0) >= 0.70
        ]


class Article(BaseModel):
    model_config = ConfigDict(extra="ignore")

    article_id: str
    published_at: Optional[str] = None
    source_name: Optional[str] = None
    title: str
    url: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    score: int = Field(ge=5, le=10)
    score_band: Literal["band_B", "band_A"]
    market_relevance_reason: str
    metadata: ArticleMetadata = Field(default_factory=ArticleMetadata)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        data = normalize_ingest_payload(data or {})
        if not isinstance(data, dict):
            return data
        metadata = dict(data.get("metadata") or {})
        for key in list(ArticleMetadata.model_fields):
            if key in data:
                metadata[key] = data.pop(key)
        data["metadata"] = metadata
        return data

    @field_validator(
        "published_at",
        "source_name",
        "url",
        "description",
        "content",
        mode="before",
    )
    @classmethod
    def clean_optional_text_fields(cls, value):
        return _clean_optional_text(value)

    @field_validator("article_id", "title", "market_relevance_reason", mode="before")
    @classmethod
    def clean_required_text(cls, value):
        return _normalize_whitespace(value)

    @model_validator(mode="after")
    def enforce_score_band_and_graph_rules(self):
        expected_band = score_to_band(self.score)
        if expected_band not in {SCORE_BAND_B, SCORE_BAND_A}:
            raise ValueError("Article can only represent band_B or band_A items")
        self.score_band = expected_band
        article_text = self.vector_text()
        self.metadata.explicit_companies_mentioned = [
            company
            for company in self.metadata.explicit_companies_mentioned
            if _company_is_directly_mentioned(company, article_text)
        ]
        return self

    @property
    def graph_facts(self) -> List[GraphFact]:
        return self.metadata.graph_facts

    def vector_text(self) -> str:
        return "\n\n".join(
            text
            for text in [self.title, self.description, self.content]
            if str(text or "").strip()
        ).strip()


class PromptBResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    articles: List[Article] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data):
        return normalize_ingest_payload(data or {})


class NegativeAnchorDecision(BaseModel):
    article_id: Optional[str] = None
    passed: bool
    max_similarity: float
    matched_negative_text: Optional[str] = None


class IngestionResult(BaseModel):
    fast_filtered: List[NegativeAnchorDecision] = Field(default_factory=list)
    scored: List[PromptAScoreResult] = Field(default_factory=list)
    extracted_articles: List[Article] = Field(default_factory=list)
    vector_count: int = 0
    graph_fact_count: int = 0
    quarantined_count: int = 0
    dropped_count: int = 0


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = list(zip(left or [], right or []))
    if not pairs:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in pairs)
    left_norm = sum(float(a) * float(a) for a, _ in pairs) ** 0.5
    right_norm = sum(float(b) * float(b) for _, b in pairs) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class NegativeAnchorFilter:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        negative_texts: Sequence[str] = NEGATIVE_TEXTS,
        threshold: float = 0.85,
        negative_anchor_vectors: Optional[Sequence[Sequence[float]]] = None,
    ):
        self.embedding_client = embedding_client
        self.negative_texts = [_normalize_whitespace(text) for text in negative_texts if text]
        self.threshold = threshold
        if negative_anchor_vectors is None and self.negative_texts:
            self.negative_anchor_vectors = list(self.embedding_client.embed_texts(self.negative_texts))
        elif negative_anchor_vectors is not None:
            self.negative_anchor_vectors = [list(vector) for vector in negative_anchor_vectors]
        else:
            self.negative_anchor_vectors = []

    def decision_for_article(self, article: RawArticle) -> NegativeAnchorDecision:
        text = article.fast_filter_text()
        if not text or not self.negative_anchor_vectors:
            return NegativeAnchorDecision(
                article_id=article.article_id,
                passed=True,
                max_similarity=0.0,
                matched_negative_text=None,
            )

        vector = list(self.embedding_client.embed_texts([text])[0])
        similarities = [
            cosine_similarity(vector, anchor_vector)
            for anchor_vector in self.negative_anchor_vectors
        ]
        max_similarity = max(similarities) if similarities else 0.0
        match_index = similarities.index(max_similarity) if similarities else -1
        matched_text = self.negative_texts[match_index] if match_index >= 0 else None
        return NegativeAnchorDecision(
            article_id=article.article_id,
            passed=max_similarity < self.threshold,
            max_similarity=max_similarity,
            matched_negative_text=matched_text,
        )

    def filter_articles(self, articles: Sequence[RawArticle]):
        passed_articles = []
        decisions = []
        for article in articles:
            decision = self.decision_for_article(article)
            decisions.append(decision)
            if decision.passed:
                passed_articles.append(article)
        return passed_articles, decisions


def _articles_json(articles: Sequence[RawArticle]) -> str:
    return json.dumps(
        [article.prompt_a_payload() for article in articles],
        ensure_ascii=False,
        sort_keys=True,
    )


def build_prompt_a(articles: Sequence[RawArticle]) -> str:
    return PROMPT_A_TEMPLATE.replace("{articles_json}", _articles_json(articles))


def run_prompt_a_scoring(
    articles: Sequence[RawArticle],
    llm_client: LLMClient,
    batch_size: int = NEWS_BATCH_SIZE,
) -> Dict[str, PromptAScoreResult]:
    results = {}
    for batch_start in range(0, len(articles), batch_size):
        batch = articles[batch_start: batch_start + batch_size]
        response = PromptAResponse.model_validate(
            _json_from_response(llm_client.complete(build_prompt_a(batch)))
        )
        for score_result in response.results:
            results[score_result.article_id] = score_result
    return results


def _merge_article_and_score(article: RawArticle, score_result: PromptAScoreResult) -> Dict[str, Any]:
    payload = article.scoring_payload()
    payload.update(
        {
            "score": score_result.score,
            "score_band": score_result.score_band,
            "market_relevance_reason": score_result.market_relevance_reason,
            "is_policy_article": getattr(score_result, "is_policy_article", False),
            "policy": (
                score_result.policy.model_dump(mode="json")
                if getattr(score_result, "policy", None) is not None
                else None
            ),
        }
    )
    return payload


def build_prompt_b(scored_articles: Sequence[Dict[str, Any]]) -> str:
    return PROMPT_B_TEMPLATE.replace(
        "{articles_json}",
        json.dumps(scored_articles, ensure_ascii=False, sort_keys=True),
    )


def run_prompt_b_extraction(
    scored_articles: Sequence[Dict[str, Any]],
    llm_client: LLMClient,
) -> Dict[str, Article]:
    if not scored_articles:
        return {}
    response = PromptBResponse.model_validate(
        _json_from_response(llm_client.complete(build_prompt_b(scored_articles)))
    )
    return {article.article_id: article for article in response.articles}


def apply_ingestion_policy(
    raw_article: RawArticle,
    score_result: PromptAScoreResult,
    vector_store: VectorStoreClient,
    graph_store: GraphStoreClient,
    quarantine_store: QuarantineStoreClient,
    article: Optional[Article] = None,
) -> Dict[str, int]:
    counts = {"vector": 0, "graph_facts": 0, "quarantine": 0, "dropped": 0}

    if score_result.score_band == SCORE_BAND_F:
        if score_result.extracted_negative_text:
            quarantine_store.append(score_result.extracted_negative_text)
            counts["quarantine"] = 1
        counts["dropped"] = 1
        return counts

    if score_result.score_band == SCORE_BAND_DROP:
        counts["dropped"] = 1
        return counts

    if article is None:
        raise ValueError("band_B and band_A articles require extracted Article metadata")

    if article.score_band == SCORE_BAND_B:
        vector_store.upsert_article(
            article=article,
            text=article.vector_text(),
            retention_months=RETENTION_MONTHS,
            permanent=False,
        )
        counts["vector"] = 1
        return counts

    if article.score_band == SCORE_BAND_A:
        vector_store.upsert_article(
            article=article,
            text=article.vector_text(),
            retention_months=None,
            permanent=True,
        )
        graph_store.upsert_graph_facts(article.graph_facts)
        counts["vector"] = 1
        counts["graph_facts"] = len(article.graph_facts)
        return counts

    counts["dropped"] = 1
    return counts


class NewsIngestionPipeline:
    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        vector_store: VectorStoreClient,
        graph_store: GraphStoreClient,
        quarantine_store: QuarantineStoreClient,
        negative_texts: Sequence[str] = NEGATIVE_TEXTS,
        negative_similarity_threshold: float = 0.85,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.quarantine_store = quarantine_store
        self.negative_filter = NegativeAnchorFilter(
            embedding_client=embedding_client,
            negative_texts=negative_texts,
            threshold=negative_similarity_threshold,
        )

    def ingest_batch(self, raw_articles: Sequence[Dict[str, Any]]) -> IngestionResult:
        articles = [RawArticle.model_validate(raw_article) for raw_article in raw_articles]
        passed_articles, filter_decisions = self.negative_filter.filter_articles(articles)
        result = IngestionResult(
            fast_filtered=[decision for decision in filter_decisions if not decision.passed],
        )
        result.dropped_count += len(result.fast_filtered)

        score_results = run_prompt_a_scoring(passed_articles, self.llm_client)
        result.scored = list(score_results.values())

        surviving_payloads = []
        surviving_article_ids = set()
        raw_by_id = {article.article_id: article for article in passed_articles}
        for score_result in result.scored:
            if score_result.score_band in {SCORE_BAND_A, SCORE_BAND_B}:
                raw_article = raw_by_id[score_result.article_id]
                surviving_payloads.append(_merge_article_and_score(raw_article, score_result))
                surviving_article_ids.add(score_result.article_id)

        extracted_by_id = run_prompt_b_extraction(surviving_payloads, self.llm_client)
        result.extracted_articles = [
            extracted_by_id[article_id]
            for article_id in surviving_article_ids
            if article_id in extracted_by_id
        ]

        for score_result in result.scored:
            raw_article = raw_by_id[score_result.article_id]
            article = extracted_by_id.get(score_result.article_id)
            counts = apply_ingestion_policy(
                raw_article=raw_article,
                score_result=score_result,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                quarantine_store=self.quarantine_store,
                article=article,
            )
            result.vector_count += counts["vector"]
            result.graph_fact_count += counts["graph_facts"]
            result.quarantined_count += counts["quarantine"]
            result.dropped_count += counts["dropped"]

        return result


def _llm_complete(prompt):
    if Settings is None:
        raise ImportError("llama_index is required for the legacy news ingestion LLM client.")
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
    score_band = score_to_band(score)
    age_days = max((now_dt - published_dt).days, 0)
    within_retention = age_days <= RETENTION_MONTHS * 30
    if score_band in {SCORE_BAND_F, SCORE_BAND_DROP}:
        return None, False
    if score_band == SCORE_BAND_A:
        return "full", True
    if score_band == SCORE_BAND_B and within_retention:
        return "full", False
    return None, False


def _save_raw_article_record(record, data_dir=DEFAULT_NEWS_DATA_DIR):
    raw_dir = _raw_article_dir(data_dir=data_dir)
    os.makedirs(raw_dir, exist_ok=True)
    filename = f"{_slugify(record['published_at'])}-{record['article_id'].split('::')[-1]}.json"
    path = os.path.join(raw_dir, filename)
    with open(path, "w", encoding="utf-8") as raw_file:
        json.dump(record, raw_file, ensure_ascii=False, indent=2, sort_keys=True)


def _world_news_request(params):
    api_key = _require_world_news_api_key()
    if requests is None:
        request = Request(
            f"{WORLD_NEWS_SEARCH_URL}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "Stock-RAG-world-news-ingest/1.0",
                "x-api-key": api_key,
            },
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "news" not in payload:
            raise RuntimeError(f"World News API request failed: {payload}")
        return payload

    response = requests.get(
        WORLD_NEWS_SEARCH_URL,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": "Stock-RAG-world-news-ingest/1.0",
            "x-api-key": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "news" not in payload:
        raise RuntimeError(f"World News API request failed: {payload}")
    return payload


def fetch_news_bucket(bucket, start_dt, end_dt, max_pages=5, page_size=100):
    query = WORLD_NEWS_BUCKET_QUERIES.get(bucket, BUCKET_QUERIES[bucket])
    articles = []
    for page in range(1, max_pages + 1):
        batch_size = min(page_size, 100)
        payload = _world_news_request(
            {
                "text": query,
                "text-match-indexes": "title,content",
                "earliest-publish-date": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "latest-publish-date": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "language": "en",
                "sort": "publish-time",
                "sort-direction": "DESC",
                "number": batch_size,
                "offset": (page - 1) * batch_size,
                "news-sources": ",".join(WORLD_NEWS_SOURCE_URLS),
            }
        )
        page_articles = payload.get("news", [])
        if not page_articles:
            break

        for raw_article in page_articles:
            url = raw_article.get("url")
            article = {
                "bucket": bucket,
                "title": _normalize_whitespace(raw_article.get("title")),
                "description": _normalize_whitespace(raw_article.get("summary")),
                "content": _normalize_whitespace(raw_article.get("text")),
                "url": url,
                "source_name": _normalize_whitespace(_source_name_from_url(url)),
                "published_at": raw_article.get("publish_date"),
            }
            article["article_id"] = _article_id(article)
            if not _source_is_allowed(article):
                continue
            if not _article_text(article):
                continue
            articles.append(article)

        if len(page_articles) < batch_size:
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
