import json
import xml.etree.ElementTree as ET

import requests

import ingest_news
import ingest_stock


BBC_BUSINESS_RSS_URL = "https://feeds.bbci.co.uk/news/business/rss.xml"


def fetch_one_general_news_article():
    response = requests.get(BBC_BUSINESS_RSS_URL, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    item = root.find("./channel/item")
    if item is None:
        raise RuntimeError("No news article found in the BBC business RSS feed.")

    def item_text(tag_name):
        node = item.find(tag_name)
        return (node.text or "").strip() if node is not None and node.text else ""

    article = {
        "article_id": "test_bbc_business_article",
        "bucket": "business",
        "title": item_text("title"),
        "description": item_text("description"),
        "content": item_text("description"),
        "url": item_text("link"),
        "source_name": "BBC News",
        "published_at": item_text("pubDate"),
    }
    return article


def main():
    ingest_stock.env()

    article = fetch_one_general_news_article()
    article_text = ingest_news._article_text(article)
    summary = ingest_news.summarize_article(article_text)
    company_aliases = ingest_news._build_company_aliases()
    metadata = ingest_news.extract_news_metadata(
        article=article,
        summary_text=summary,
        article_text=article_text,
        company_aliases=company_aliases,
    )

    print("TITLE:")
    print(article["title"])
    print()
    print("SOURCE:")
    print(article["source_name"])
    print()
    print("SUMMARY:")
    print(summary)
    print()
    print("METADATA:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
