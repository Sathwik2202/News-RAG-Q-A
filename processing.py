"""
Chunking + deduplication.

News articles are short, so we use light chunking (not aggressive splitting),
and dedup on embedding similarity since wire stories get republished
near-verbatim across many outlets.
"""

from dataclasses import dataclass
import numpy as np

from config import CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, DEDUP_SIMILARITY_THRESHOLD
from ingest import Article


@dataclass
class Chunk:
    id: str
    text: str
    article_id: str
    title: str
    source: str
    url: str
    published: str


def chunk_article(article: Article) -> list[Chunk]:
    """Split article text into overlapping chunks. Short articles stay as one chunk."""
    text = article.text.strip()

    if len(text) <= CHUNK_SIZE_CHARS:
        return [
            Chunk(
                id=f"{article.id}_0",
                text=text,
                article_id=article.id,
                title=article.title,
                source=article.source,
                url=article.url,
                published=article.published or "",
            )
        ]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        piece = text[start:end]
        chunks.append(
            Chunk(
                id=f"{article.id}_{idx}",
                text=piece,
                article_id=article.id,
                title=article.title,
                source=article.source,
                url=article.url,
                published=article.published or "",
            )
        )
        start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS  # <--- FIXED HERE
        idx += 1

    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def dedup_articles(articles: list[Article], embed_fn) -> list[Article]:
    """
    Drop near-duplicate articles (e.g. the same AP/Reuters wire story
    republished across outlets). Keeps the first-seen copy.

    embed_fn: a function taking list[str] -> list[list[float]]
    """
    if not articles:
        return []

    # Embed on title + first 300 chars, which is enough to catch republished wire stories
    fingerprints = [f"{a.title}\n{a.text[:300]}" for a in articles]
    vectors = np.array(embed_fn(fingerprints))

    keep: list[Article] = []
    kept_vectors: list[np.ndarray] = []

    for article, vec in zip(articles, vectors):
        is_dup = False
        for kv in kept_vectors:
            if cosine_sim(vec, kv) >= DEDUP_SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            keep.append(article)
            kept_vectors.append(vec)

    print(f"[dedup] {len(articles)} -> {len(keep)} after removing near-duplicates")
    return keep