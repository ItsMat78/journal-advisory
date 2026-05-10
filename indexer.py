"""
indexer.py — Scans the Obsidian vault, chunks markdown files, embeds them
with sentence-transformers, and stores everything in a local ChromaDB.

Run directly to re-index the full vault:
    python indexer.py
"""

import contextlib
import hashlib
import io
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for non-ASCII chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DB_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
    LOG_FILE,
    get_vault_path,
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Month name → zero-padded number ──────────────────────────────────────────
_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_embedding_model() -> SentenceTransformer:
    # Redirect stdout + stderr to swallow library noise during model loading:
    # BertModel LOAD REPORT (sentence-transformers 5.x print()) and the
    # HuggingFace Hub unauthenticated-request warning (direct stderr write).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return SentenceTransformer(EMBEDDING_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_path(p: Path, vault: Path) -> str:
    """Return a forward-slash relative path string (ChromaDB where-clause safe)."""
    return str(p.relative_to(vault)).replace("\\", "/")


def parse_date_from_path(filepath: Path, vault_path: Path) -> Optional[str]:
    """
    Attempt to extract a YYYY-MM-DD date from the file path.

    Strategies tried in order:
      1. YYYY-MM-DD anywhere in the stem
      2. DD-MM-YYYY in the stem
      3. "Month DD" or "Month DD, YYYY" in the stem
      4. Parent folder is a month name → folder + day from stem + mtime year
    """
    stem = filepath.stem

    # 1. ISO date in stem: 2024-01-15
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 2. DD-MM-YYYY or DD.MM.YYYY
    m = re.match(r'(\d{1,2})[-.](\d{1,2})[-.](\d{4})', stem)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    # 3. Ordinal day + month + year: "8th May 2026", "1st January 2025"
    #    Must come before the bare "Month DD" pattern below — otherwise "May 2026"
    #    in "8th May 2026" greedily matches as month=May, day=20 (first 2 chars of 2026).
    m = re.search(
        r'(\d{1,2})(?:st|nd|rd|th)\s+'
        r'(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+(\d{4})',
        stem, re.I
    )
    if m:
        day, month_str, year = m.groups()
        month = _MONTH_MAP[month_str.lower()]
        return f"{year}-{month}-{day.zfill(2)}"

    # 4. Month name + bare day (no ordinal): "January 15, 2024" or "January 15"
    m = re.search(
        r'(january|february|march|april|may|june|july|august|'
        r'september|october|november|december)\s+(\d{1,2})(?:[,\s]+(\d{4}))?',
        stem, re.I
    )
    if m:
        month_str, day, year = m.groups()
        month = _MONTH_MAP[month_str.lower()]
        if not year:
            year = str(datetime.fromtimestamp(filepath.stat().st_mtime).year)
        return f"{year}-{month}-{day.zfill(2)}"

    # 4. Parent folder is a month name (Daily/January/15.md)
    parts = list(filepath.relative_to(vault_path).parts)
    if len(parts) >= 2:
        parent_lower = parts[-2].lower()
        month = _MONTH_MAP.get(parent_lower)
        if month:
            day_m = re.search(r'(\d{1,2})', stem)
            if day_m:
                year = str(datetime.fromtimestamp(filepath.stat().st_mtime).year)
                return f"{year}-{month}-{day_m.group(1).zfill(2)}"

    return None


def classify_file_type(filepath: Path, vault_path: Path) -> str:
    parts = list(filepath.relative_to(vault_path).parts)
    if not parts:
        return "note"
    top = parts[0].lower()
    if top == "daily":
        return "daily"
    if top == "people":
        return "people"
    return "note"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleaning — strip Obsidian syntax before chunking
# ─────────────────────────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """
    Strip Obsidian-specific syntax so only plain prose reaches the embedder.

    Steps:
      1. Remove any residual YAML frontmatter blocks (--- ... ---)
      2. Remove image embeds: ![[img]] and ![alt](url)
      3. Wikilinks: [[Page|display]] → display, [[Page]] → Page
      4. Markdown links: [text](url) → text
      5. Obsidian #tags (not markdown # headings)
      6. HTML tags
      7. Collapse excess blank lines
    """
    # 1. Residual frontmatter (safety net — python-frontmatter already strips it)
    text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)

    # 2. Image embeds
    text = re.sub(r'!\[\[.*?\]\]', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 3. Wikilinks: [[Page|display]] → display; [[Page]] → Page
    text = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]', r'\1', text)

    # 4. Markdown links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 5. Obsidian #tags — match #word (no space after #) but not # Heading
    #    Pattern: a # not preceded by a word char, followed immediately by a letter
    text = re.sub(r'(?<!\w)#([a-zA-Z][a-zA-Z0-9_/\-]*)', '', text)

    # 6. HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 7. Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-boundary chunking
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences at sentence-ending punctuation.
    Handles common abbreviations imperfectly but well enough for journal prose.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* words,
    breaking at sentence boundaries instead of mid-word.

    Paragraphs are split into sentences first; sentences from all paragraphs
    are then packed greedily into chunks. A chunk may span paragraph
    boundaries when sentences from consecutive paragraphs fit within the limit.
    """
    # Collect sentences from each paragraph in order
    raw_sentences: list[str] = []
    for para in re.split(r'\n\n+', text):
        para = para.strip()
        if para:
            raw_sentences.extend(_split_into_sentences(para))

    if not raw_sentences:
        return []

    chunks: list[str] = []
    cur_sents: list[str] = []
    cur_words = 0

    for sent in raw_sentences:
        sent_words = len(sent.split())

        # Flush current chunk when adding this sentence would overflow
        if cur_words + sent_words > chunk_size and cur_sents:
            chunks.append(" ".join(cur_sents))

            # Keep a tail of sentences as overlap for the next chunk
            overlap_sents: list[str] = []
            overlap_words = 0
            for s in reversed(cur_sents):
                w = len(s.split())
                if overlap_words + w > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_words += w

            cur_sents  = overlap_sents
            cur_words  = overlap_words

        cur_sents.append(sent)
        cur_words += sent_words

    if cur_sents:
        chunks.append(" ".join(cur_sents))

    return chunks


def _make_chunk_id(rel_path_str: str, chunk_index: int) -> str:
    """Stable, ChromaDB-safe ID for a chunk."""
    raw = f"{rel_path_str}::chunk_{chunk_index}"
    return hashlib.sha1(raw.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Mtime helpers — skip unchanged files on re-index
# ─────────────────────────────────────────────────────────────────────────────

def get_stored_mtime(filepath: Path, vault_path: Path, collection) -> str | None:
    """
    Return the mtime string stored in ChromaDB for *filepath*, or None
    if the file has not been indexed yet (or has no mtime metadata).
    """
    rel_path_str = _norm_path(filepath, vault_path)
    try:
        results = collection.get(
            where={"source": rel_path_str},
            include=["metadatas"],
        )
        if results["metadatas"]:
            return results["metadatas"][0].get("mtime") or None
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core indexing
# ─────────────────────────────────────────────────────────────────────────────

def index_file(
    filepath: Path,
    vault_path: Path,
    collection,
    model: SentenceTransformer,
    force: bool = False,
) -> int:
    """
    Index a single markdown file into ChromaDB.

    Returns:
      -1  — skipped because mtime matches stored value (force=False only)
       0  — nothing indexed (file empty, parse error, or no text after cleaning)
       N  — number of new chunks stored
    """
    # ── Mtime check — skip files that haven't changed ──────────────────────
    current_mtime = str(filepath.stat().st_mtime)
    if not force:
        stored = get_stored_mtime(filepath, vault_path, collection)
        if stored and stored == current_mtime:
            logger.debug(f"Unchanged: {filepath.name}")
            return -1

    # ── Remove stale chunks before re-indexing ─────────────────────────────
    # (handles chunk-count changes — upsert alone would leave ghost chunks)
    remove_file_from_index(filepath, vault_path, collection)

    try:
        post    = frontmatter.load(str(filepath))
        content = _clean_markdown(post.content)
        fm      = post.metadata

        if not content:
            logger.info(f"Skip (empty after cleaning): {filepath.name}")
            return 0

        # ── Date ──────────────────────────────────────────────────────────
        date = fm.get("date") or fm.get("created") or fm.get("Date")
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")
        elif date:
            date = str(date)[:10]
        else:
            date = parse_date_from_path(filepath, vault_path)

        if not date:
            date = datetime.fromtimestamp(filepath.stat().st_mtime).strftime("%Y-%m-%d")

        rel_path_str = _norm_path(filepath, vault_path)
        file_type    = classify_file_type(filepath, vault_path)

        # ── Chunk + embed ──────────────────────────────────────────────────
        chunks = chunk_text(content)
        if not chunks:
            return 0

        embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32).tolist()

        ids       = [_make_chunk_id(rel_path_str, i) for i in range(len(chunks))]
        # date_int stores the date as YYYYMMDD integer for ChromaDB range queries
        # ($gte/$lte only work on numeric fields, not strings)
        try:
            date_int = int(date.replace("-", ""))
        except (ValueError, AttributeError):
            date_int = 0

        metadatas = [
            {
                "source":      rel_path_str,
                "date":        date,
                "date_int":    date_int,
                "file_type":   file_type,
                "chunk_index": i,
                "filename":    filepath.name,
                "mtime":       current_mtime,
                "summarised":  False,
            }
            for i in range(len(chunks))
        ]

        collection.upsert(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )
        logger.info(
            f"Indexed '{filepath.name}' -> {len(chunks)} chunks [{file_type}, {date}]"
        )
        return len(chunks)

    except Exception as exc:
        logger.error(f"Error indexing {filepath}: {exc}", exc_info=True)
        return 0


def remove_file_from_index(filepath: Path, vault_path: Path, collection) -> None:
    """Delete all ChromaDB entries that belong to this file."""
    rel_path_str = _norm_path(filepath, vault_path)
    try:
        results = collection.get(where={"source": rel_path_str})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            logger.info(f"Removed {len(results['ids'])} chunks for '{filepath.name}'")
    except Exception as exc:
        logger.error(f"Error removing '{filepath}' from index: {exc}", exc_info=True)


def cleanup_orphans(vault_path: Path, collection) -> int:
    """
    Remove index entries for files that no longer exist in the vault.
    Returns the count of source files whose entries were deleted.
    """
    try:
        results = collection.get(include=["metadatas"])
    except Exception as exc:
        logger.error(f"Orphan cleanup failed: {exc}")
        return 0

    if not results["metadatas"]:
        return 0

    to_delete: dict[str, list[str]] = {}
    for doc_id, meta in zip(results["ids"], results["metadatas"]):
        source = meta.get("source")
        if source and not (vault_path / source).exists():
            to_delete.setdefault(source, []).append(doc_id)

    for source, ids in to_delete.items():
        try:
            collection.delete(ids=ids)
            logger.info(f"Orphan removed: '{source}' ({len(ids)} chunks)")
        except Exception as exc:
            logger.error(f"Failed to delete orphan '{source}': {exc}")

    return len(to_delete)


# ─────────────────────────────────────────────────────────────────────────────
# Vault scan
# ─────────────────────────────────────────────────────────────────────────────

def scan_vault(vault_path: Path) -> list[Path]:
    """Return all indexable markdown files in the vault."""
    files: list[Path] = []
    for root, dirs, filenames in os.walk(vault_path):
        root_path = Path(root)

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        rel_root = root_path.relative_to(vault_path)
        if any(p.lower() == "images" for p in rel_root.parts):
            dirs.clear()
            continue

        for filename in filenames:
            fp = root_path / filename
            if fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(fp)

    return sorted(files)


# ─────────────────────────────────────────────────────────────────────────────
# Full-vault index
# ─────────────────────────────────────────────────────────────────────────────

def index_vault(vault_path: Path | None = None, force: bool = False) -> dict:
    """
    Index the entire vault.

    Returns {'files': N, 'chunks': M, 'skipped': K} where 'skipped' counts
    files whose mtime matched what was already stored (no work needed).
    Pass force=True to re-index every file regardless of mtime.
    """
    if vault_path is None:
        vp_str = get_vault_path()
        if not vp_str:
            sys.exit("Vault path not configured. Run setup.py first.")
        vault_path = Path(vp_str)

    if not vault_path.is_dir():
        sys.exit(f"Vault path does not exist: {vault_path}")

    print(f"Loading embedding model ({EMBEDDING_MODEL}) ...")
    model = load_embedding_model()

    client     = get_chroma_client()
    collection = get_collection(client)

    files = scan_vault(vault_path)
    print(f"Found {len(files)} markdown files.")

    total_chunks = 0
    skipped      = 0
    for i, fp in enumerate(files, 1):
        print(f"  [{i:>4}/{len(files)}] {fp.name[:60]}", end="\r", flush=True)
        result = index_file(fp, vault_path, collection, model, force=force)
        if result == -1:
            skipped += 1
        elif result > 0:
            total_chunks += result

    indexed = len(files) - skipped
    print(
        f"\nDone: {indexed} files indexed ({skipped} unchanged), "
        f"{total_chunks} chunks stored in ChromaDB."
    )
    logger.info(
        f"Full index complete: {indexed} indexed, {skipped} skipped, "
        f"{total_chunks} chunks"
    )

    orphans = cleanup_orphans(vault_path, collection)
    if orphans:
        print(f"Removed {orphans} orphaned file(s) from index.")
        logger.info(f"Orphan cleanup: {orphans} file(s) removed.")

    return {"files": len(files), "chunks": total_chunks, "skipped": skipped}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index the Obsidian vault into ChromaDB")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-index all files even if mtime is unchanged"
    )
    args = parser.parse_args()
    index_vault(force=args.force)
