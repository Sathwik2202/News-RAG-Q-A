"""
CLI entrypoint.

Usage:
    python main.py ingest              # pull latest news, index it
    python main.py ask "your question" # query the index
    python main.py purge               # drop stale articles
"""

import sys

from config import MAX_ARTICLE_AGE_DAYS
from ingest import ingest_all
from processing import chunk_article, dedup_articles
from vector_store import NewsVectorStore
from rag import answer_question


def run_ingest():
    store = NewsVectorStore()

    articles = ingest_all()
    articles = dedup_articles(articles, store.embed)

    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article))

    store.upsert_chunks(all_chunks)
    print(f"[main] ingestion complete: {len(articles)} articles, {len(all_chunks)} chunks")


def run_ask(question: str):
    store = NewsVectorStore()
    result = answer_question(question, store)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    for s in result["sources"]:
        print(f"- {s['title']} ({s['source']}, {s['published'][:10] if s['published'] else 'n/a'})")
        print(f"  {s['url']}")


def run_purge():
    store = NewsVectorStore()
    store.purge_older_than(MAX_ARTICLE_AGE_DAYS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "ingest":
        run_ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: python main.py ask \"your question\"")
            sys.exit(1)
        run_ask(sys.argv[2])
    elif cmd == "purge":
        run_purge()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
