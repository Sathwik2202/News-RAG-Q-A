# News RAG

A retrieval-augmented pipeline for asking questions about current news,
grounded in freshly ingested articles with source citations.

## How it works

```
RSS feeds -> extract clean text -> dedup near-identical wire stories
          -> chunk -> embed -> store in Chroma (with date/source metadata)
          -> at query time: semantic search + date/source filters
          -> LLM answers using only retrieved excerpts, with citations
```

## Setup

```bash
cd news_rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set an API key for whichever LLM backend you use in `config.py`:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

`EMBEDDING_BACKEND = "local"` (the default) needs no API key — it uses
`sentence-transformers` and runs on your machine.

## Usage

**1. Ingest the latest news** (run this regularly — cron every 15-60 min for a
live index, or manually whenever you want fresh data):

```bash
python main.py ingest
```

**2. Ask a question:**

```bash
python main.py ask "What's the latest on the trade negotiations?"
```

This retrieves the most relevant, recent chunks (default: last 7 days) and
asks the LLM to answer using only those excerpts, with citations, and a
staleness warning if the retrieved info looks out of date for the question.

**3. Clean up old articles** (drop anything older than `MAX_ARTICLE_AGE_DAYS`
in `config.py`, default 30 days):

```bash
python main.py purge
```

## Automating ingestion

Add a cron entry to keep the index fresh:

```bash
*/30 * * * * cd /path/to/news_rag && venv/bin/python main.py ingest >> ingest.log 2>&1
```

## Customizing

- **Add sources:** edit `RSS_FEEDS` in `config.py`. Any valid RSS/Atom feed
  works.
- **Adjust freshness window per query:** pass `days_back=` to
  `answer_question()` in `rag.py`, or add a CLI flag in `main.py`.
- **Switch embedding/LLM backends:** change `EMBEDDING_BACKEND` /
  `LLM_BACKEND` in `config.py`.
- **Tune dedup strictness:** `DEDUP_SIMILARITY_THRESHOLD` in `config.py`
  (lower = more aggressive deduping).

## File overview

| File | Purpose |
|---|---|
| `config.py` | Sources, model choices, thresholds |
| `ingest.py` | Pull RSS feeds, extract clean article text |
| `processing.py` | Chunking + near-duplicate detection |
| `vector_store.py` | Chroma wrapper, embeddings, filtered search |
| `rag.py` | Retrieval + cited answer generation |
| `main.py` | CLI: `ingest`, `ask`, `purge` |

## Known limitations (good next steps)

- Paywalled articles are skipped, not summarized from snippets
- Dedup fingerprints on title + first 300 chars — very reworded wire
  stories may slip through
- No incremental "what changed since last ingest" diff — each ingest
  just upserts by URL hash, so re-running is idempotent but not smart
  about deltas
