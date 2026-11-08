import os
import httpx  # Standard library HTTP client used by OpenAI
from openai import OpenAI
from config import LLM_BACKEND, GROQ_MODEL
from vector_store import NewsVectorStore

SYSTEM_PROMPT = """You are a news assistant. Answer the user's question using ONLY \
the provided article excerpts. Follow these rules:

1. Cite every claim with the source and date in brackets, e.g. [BBC News, 2026-08-05].
2. If the excerpts don't contain enough information to answer, say so plainly.
3. If the excerpts seem outdated relative to the question (e.g. asking about \
"today" but sources are from days ago), flag that explicitly.
4. Do not use outside knowledge. Do not speculate beyond what's in the excerpts.
5. Be concise and neutral in tone.
"""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        date = c.get("published", "unknown date")[:10] if c.get("published") else "unknown date"
        blocks.append(
            f"[{c.get('source', 'Unknown')}, {date}] {c.get('title', 'Untitled')}\n"
            f"URL: {c.get('url', '')}\n{c.get('text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _call_groq(question: str, context: str) -> str:
    # Explicitly create an httpx client with trust_env=False to ignore Windows proxy settings
    http_client = httpx.Client(trust_env=False)

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        http_client=http_client,
    )
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Article excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def answer_question(
    question: str,
    store: NewsVectorStore,
    top_k: int = 6,
    days_back: int | None = 7,
    source: str | None = None,
) -> dict:
    chunks = store.query(question, top_k=top_k, days_back=days_back, source=source)

    if not chunks:
        return {
            "answer": "No relevant articles found in the index for this question.",
            "sources": [],
        }

    context = build_context(chunks)

    if LLM_BACKEND == "groq":
        answer = _call_groq(question, context)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

    return {
        "answer": answer,
        "sources": [
            {
                "title": c.get("title", ""),
                "source": c.get("source", ""),
                "url": c.get("url", ""),
                "published": c.get("published", ""),
            }
            for c in chunks
        ],
    }