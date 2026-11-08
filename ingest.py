"""
Ingestion layer: pull articles from RSS feeds, extract clean text.

Run standalone:  python ingest.py
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import trafilatura
from dateutil import parser as dateparser

from config import RSS_FEEDS


@dataclass
class Article:
    url: str
    title: str
    source: str
    published: Optional[str]  # ISO 8601 string
    text: str
    id: str = field(init=False)

    def __post_init__(self):
        # Stable id from URL so re-ingesting the same article is idempotent
        self.id = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:24]


def _parse_date(entry) -> Optional[str]:
    for key in ("published", "updated", "pubDate"):
        val = entry.get(key)
        if val:
            try:
                return dateparser.parse(val).isoformat()
            except (ValueError, TypeError):
                continue
    return None


def fetch_feed_entries(feed_url: str) -> list[dict]:
    parsed = feedparser.parse(feed_url)
    source = parsed.feed.get("title", feed_url)
    entries = []
    for entry in parsed.entries:
        link = entry.get("link")
        if not link:
            continue
        entries.append(
            {
                "url": link,
                "title": entry.get("title", "").strip(),
                "source": source,
                "published": _parse_date(entry),
            }
        )
    return entries


def extract_article_text(url: str) -> Optional[str]:
    """Download and extract clean article body text, stripping ads/nav/boilerplate."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return text


def ingest_all(feeds: list[str] = None, delay: float = 0.5) -> list[Article]:
    """Fetch every feed, extract full text for each entry. Returns Article objects."""
    feeds = feeds or RSS_FEEDS
    articles: list[Article] = []

    for feed_url in feeds:
        print(f"[ingest] fetching feed: {feed_url}")
        try:
            entries = fetch_feed_entries(feed_url)
        except Exception as e:
            print(f"[ingest] failed to parse feed {feed_url}: {e}")
            continue

        for entry in entries:
            text = extract_article_text(entry["url"])
            if not text or len(text) < 200:
                # Too short = likely paywall, extraction failure, or video-only page
                continue

            articles.append(
                Article(
                    url=entry["url"],
                    title=entry["title"],
                    source=entry["source"],
                    published=entry["published"],
                    text=text,
                )
            )
            time.sleep(delay)  # be polite to servers

        print(f"[ingest] got {len(entries)} entries from {feed_url}")

    print(f"[ingest] total articles extracted: {len(articles)}")
    return articles


if __name__ == "__main__":
    arts = ingest_all()
    for a in arts[:5]:
        print(f"- {a.title[:70]!r} ({a.source}, {a.published})")
