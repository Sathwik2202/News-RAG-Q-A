"""
Central config for the News RAG pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root and sets env vars from it

# --- RSS sources (add/remove freely) ---
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "http://rss.cnn.com/rss/cnn_topstories.rss",
]

# --- Storage ---
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "news_articles"

# --- Embeddings (local, free, no API key) ---
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- LLM for generation (needs GEMINI_API_KEY in .env) ---
LLM_BACKEND = "groq"
GROQ_MODEL = "openai/gpt-oss-120b"

# --- Dedup ---
DEDUP_SIMILARITY_THRESHOLD = 0.93  # cosine similarity above this = duplicate

# --- Freshness ---
MAX_ARTICLE_AGE_DAYS = 30  # drop articles older than this on cleanup

# --- Chunking ---
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150