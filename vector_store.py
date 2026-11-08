"""
Vector store wrapper (Chroma) + local embedding model.

Handles: embedding text, upserting chunks with metadata, and
metadata-filtered semantic search (e.g. "last 7 days" + source filter).
"""

import os
from datetime import datetime, timedelta, timezone

import chromadb
from dateutil import parser as dateparser

from config import DB_DIR, COLLECTION_NAME, LOCAL_EMBEDDING_MODEL
from processing import Chunk
os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Silences Chroma telemetry warning

_local_model = None


def get_embedder():
    """Returns a function: list[str] -> list[list[float]] (runs locally, free)."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    def embed(texts: list[str]) -> list[list[float]]:
        return _local_model.encode(texts, show_progress_bar=False).tolist()

    return embed


def _to_ts(iso: str) -> int:
    """ISO date string -> unix timestamp (0 if missing or unparseable).

    Chroma's range filters ($gte, $lt) only work on numbers, so we store
    a numeric timestamp next to the human-readable date string.
    """
    if not iso:
        return 0
    try:
        dt = dateparser.parse(iso)
    except (ValueError, TypeError, OverflowError):
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _cutoff_ts(days: int) -> int:
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())


class NewsVectorStore:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        self.embed = get_embedder()

    def upsert_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = self.embed([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "article_id": c.article_id,
                    "title": c.title or "Untitled",
                    "source": c.source or "Unknown Source",
                    "url": c.url or "",
                    "published": c.published or "",
                    "published_ts": _to_ts(c.published),
                }
                for c in chunks
            ],
        )
        print(f"[store] upserted {len(chunks)} chunks")

    def query(
        self,
        query_text: str,
        top_k: int = 6,
        days_back: int | None = None,
        source: str | None = None,
    ):
        """Semantic search with optional recency + source metadata filters."""
        vec = self.embed([query_text])[0]

        where_clauses = []
        if source:
            where_clauses.append({"source": source})
        if days_back is not None:
            where_clauses.append({"published_ts": {"$gte": _cutoff_ts(days_back)}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        results = self.collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            where=where,
        )
        return self._format_results(results)

    @staticmethod
    def _format_results(results) -> list[dict]:
        out = []
        if not results or not results.get("documents"):
            return out
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({**meta, "text": doc, "distance": dist})
        return out

    def purge_older_than(self, days: int):
        """Drop stale articles from the index."""
        cutoff = _cutoff_ts(days)
        self.collection.delete(where={"published_ts": {"$lt": cutoff}})
        print(f"[store] purged chunks published before "
              f"{datetime.fromtimestamp(cutoff, timezone.utc).isoformat()}")