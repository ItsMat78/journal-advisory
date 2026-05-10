"""
config.py — Central configuration for journal-rag.
All paths are relative to this file's parent directory.
"""
import json
import logging
import os
import warnings
from pathlib import Path

# Suppress noisy output from HuggingFace / transformers / tqdm before those
# libraries are imported. config.py is always the first project module loaded,
# so setting these here guarantees they're in place before sentence-transformers
# (and tqdm) are imported by retriever.py or indexer.py.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # model is cached; skip Hub checks + auth warning

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
DB_DIR      = BASE_DIR / "db"
TEMP_DIR    = BASE_DIR / "tmp"
LOG_FILE    = BASE_DIR / "journal-rag.log"
CONFIG_FILE = BASE_DIR / "config.json"

# ── Embedding model (local, no API) ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "journal_entries"

# ── Claude models ─────────────────────────────────────────────────────────────
# Default: Sonnet for everyday questions (Pro subscription)
MODEL_DEFAULT = "claude-sonnet-4-6"
# Deep: Opus for deeper analysis; activated with --deep flag (requires Max)
MODEL_DEEP    = "claude-opus-4-6"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 300   # max words per chunk
CHUNK_OVERLAP = 30    # overlap words between consecutive chunks (≈1 sentence)

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K                = 15    # final chunks returned to Claude
RETRIEVAL_OVERSAMPLE = 3     # fetch TOP_K × this many candidates before reranking

# ── Recency weighting ─────────────────────────────────────────────────────────
# Hybrid score = (1 - RECENCY_WEIGHT) * semantic + RECENCY_WEIGHT * recency
# Recency score uses exponential decay: 1.0 today → 0.5 at RECENCY_HALF_LIFE_DAYS
RECENCY_WEIGHT         = 0.3
RECENCY_HALF_LIFE_DAYS = 30.0

# ── Vault scanning ────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".md"}

# Directory names (exact, case-sensitive) to skip while walking the vault
SKIP_DIRS = {"Images", "images", ".git", ".obsidian", ".trash"}

# ── Summarisation ─────────────────────────────────────────────────────────────
# Daily entries older than this many days are candidates for compression
SUMMARY_AGE_DAYS = 90
# Anthropic model used by summarise_old.py
SUMMARY_MODEL = "claude-haiku-4-5-20251001"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_vault_path() -> str | None:
    """Return the vault path string from config, or None if not set."""
    return load_config().get("vault_path")
