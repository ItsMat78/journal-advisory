"""
retriever.py — Embeds a query and returns the top-k most relevant journal
chunks from ChromaDB using a hybrid semantic + recency score.
"""

import math
import subprocess
import sys
from datetime import date
from sentence_transformers import SentenceTransformer

from config import (
    COLLECTION_NAME, EMBEDDING_MODEL, TOP_K,
    RECENCY_WEIGHT, RECENCY_HALF_LIFE_DAYS, RETRIEVAL_OVERSAMPLE,
    get_vault_path,
)
from indexer import get_chroma_client, get_collection, load_embedding_model

# ─────────────────────────────────────────────────────────────────────────────
# Query rewriting
# ─────────────────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = """\
You are a search query optimizer for a personal journal retrieval system that uses \
semantic similarity search.

Rewrite the user's question into a clean retrieval query. Rules:
- Keep only what SHOULD be found — strip negations entirely (if they say \
"not about exams", drop "exams" from the query)
- Remove meta-instructions: "analyse", "go through", "tell me", "compare", \
"what patterns", etc.
- Expand with synonyms or related terms where it helps (e.g. "stress" → \
"stress anxiety pressure overwhelmed burnout")
- Output ONLY the rewritten query. One or two lines. No explanation.

Question: {question}
Retrieval query:"""


def _rewrite_query(question: str) -> str:
    """
    Use a fast model to produce a retrieval-optimised version of the question.
    Falls back to the original question if the call fails or times out.
    """
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "claude-haiku-4-5-20251001"],
            input=_REWRITE_PROMPT.format(question=question),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
        )
        if result.returncode == 0:
            rewritten = result.stdout.strip()
            if rewritten:
                return rewritten
    except Exception:
        pass
    return question


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedChunk:
    """A single result chunk with its text and metadata."""

    __slots__ = ("text", "source", "date", "file_type", "chunk_index",
                 "distance", "hybrid_score")

    def __init__(
        self,
        text: str,
        source: str,
        date: str,
        file_type: str,
        chunk_index: int,
        distance: float,
        hybrid_score: float = 0.0,
    ):
        self.text         = text
        self.source       = source
        self.date         = date
        self.file_type    = file_type
        self.chunk_index  = chunk_index
        self.distance     = distance
        self.hybrid_score = hybrid_score

    @property
    def score(self) -> float:
        """Pure semantic similarity (higher = more similar), range 0-1."""
        return max(0.0, 1.0 - self.distance)

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(source={self.source!r}, date={self.date!r}, "
            f"hybrid={self.hybrid_score:.3f}, text={self.text[:60]!r}…)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Recency scoring
# ─────────────────────────────────────────────────────────────────────────────

def _recency_score(date_str: str, today: date, half_life_days: float) -> float:
    """
    Exponential decay: 1.0 for today, 0.5 at half_life_days, ~0.12 at 3×half_life.
    Returns 0.5 (neutral) if the date can't be parsed.
    """
    try:
        entry_date = date.fromisoformat(date_str[:10])
        days_ago   = max(0, (today - entry_date).days)
        return math.exp(-math.log(2) * days_ago / half_life_days)
    except (ValueError, TypeError):
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Retriever class (lazy-loads model + DB on first use)
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._collection = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = load_embedding_model()
        if self._collection is None:
            client           = get_chroma_client()
            self._collection = get_collection(client)

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        since: str | None = None,
        until: str | None = None,
        long_term_only: bool = False,
        recency_weight: float = RECENCY_WEIGHT,
    ) -> list[RetrievedChunk]:
        """
        Embed *question* and return the *top_k* best chunks by hybrid score.

        Hybrid score = (1 - recency_weight) * semantic + recency_weight * recency.
        Oversample by RETRIEVAL_OVERSAMPLE × top_k candidates from ChromaDB,
        rerank by hybrid score, then return the top top_k.

        Args:
            since: optional ISO date string (YYYY-MM-DD) — only return chunks
                   from entries on or after this date.
            until: optional ISO date string (YYYY-MM-DD) — only return chunks
                   from entries on or before this date.
            long_term_only: if True, restrict search to entries older than the
                   current calendar month plus all people notes (entries from
                   this month are handled by short-term memory in the caller).
        """
        self._ensure_loaded()

        retrieval_query = _rewrite_query(question)
        q_embedding     = self._model.encode(retrieval_query, show_progress_bar=False).tolist()
        total_docs      = self._collection.count()
        if total_docs == 0:
            return []

        n_candidates = max(top_k, min(top_k * RETRIEVAL_OVERSAMPLE, total_docs))

        if long_term_only:
            month_start_int = int(date.today().strftime("%Y%m01"))
            where: dict | None = {"$or": [
                {"date_int": {"$lt": month_start_int}},
                {"file_type": {"$eq": "people"}},
            ]}
        elif since or until:
            conditions = []
            if since:
                conditions.append({"date_int": {"$gte": int(since.replace("-", ""))}})
            if until:
                conditions.append({"date_int": {"$lte": int(until.replace("-", ""))}})
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        else:
            where = None

        try:
            results = self._collection.query(
                query_embeddings=[q_embedding],
                n_results=n_candidates,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[retriever] ChromaDB query failed: {exc}", file=sys.stderr)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        today  = date.today()
        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sem    = max(0.0, 1.0 - float(dist))
            rec    = _recency_score(meta.get("date", ""), today, RECENCY_HALF_LIFE_DAYS)
            hybrid = (1.0 - recency_weight) * sem + recency_weight * rec
            chunks.append(
                RetrievedChunk(
                    text         = doc,
                    source       = meta.get("source", "unknown"),
                    date         = meta.get("date", "unknown"),
                    file_type    = meta.get("file_type", "note"),
                    chunk_index  = int(meta.get("chunk_index", 0)),
                    distance     = float(dist),
                    hybrid_score = hybrid,
                )
            )

        chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
        return chunks[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_retriever = Retriever()


def retrieve(
    question: str,
    top_k: int = TOP_K,
    since: str | None = None,
    until: str | None = None,
    long_term_only: bool = False,
) -> list[RetrievedChunk]:
    """Convenience function — uses the module-level Retriever singleton."""
    return _retriever.query(question, top_k=top_k, since=since, until=until, long_term_only=long_term_only)


def retrieve_current_month() -> list[RetrievedChunk]:
    """
    Return all non-people chunks from the current calendar month, sorted
    chronologically. Used as short-term memory — always present in context.
    """
    month_start_int = int(date.today().strftime("%Y%m01"))
    _retriever._ensure_loaded()
    try:
        data = _retriever._collection.get(
            where={"$and": [
                {"date_int": {"$gte": month_start_int}},
                {"file_type": {"$ne": "people"}},
            ]},
            include=["metadatas", "documents"],
        )
    except Exception as exc:
        print(f"[retriever] retrieve_current_month failed: {exc}", file=sys.stderr)
        return []

    if not data["documents"]:
        return []

    chunks = [
        RetrievedChunk(
            text        = doc,
            source      = meta.get("source", "unknown"),
            date        = meta.get("date", "unknown"),
            file_type   = meta.get("file_type", "note"),
            chunk_index = int(meta.get("chunk_index", 0)),
            distance    = 0.0,
            hybrid_score= 1.0,
        )
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]
    chunks.sort(key=lambda c: (c.date, c.source, c.chunk_index))
    return chunks


def retrieve_latest(
    n_entries: int = 1,
    history_top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    """
    Return ALL chunks from the N most recently dated source files (complete,
    in reading order) followed by TOP_K semantically relevant chunks from
    older entries. The historical context is derived by querying the vault
    with the text of the latest entries as the search signal, so it reflects
    the same themes and dynamics — giving Claude full current entries plus
    the history needed to identify patterns and advise well.
    """
    _retriever._ensure_loaded()

    all_data = _retriever._collection.get(include=["metadatas", "documents"])
    if not all_data["metadatas"]:
        return []

    # Find the most recent date per unique source file
    latest_date_per_source: dict[str, str] = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "")
        d   = meta.get("date", "")
        if src and d:
            if src not in latest_date_per_source or d > latest_date_per_source[src]:
                latest_date_per_source[src] = d

    # Pick the N most recent source files
    top_sources = {
        src for src, _ in sorted(
            latest_date_per_source.items(), key=lambda x: x[1], reverse=True
        )[:n_entries]
    }

    # All chunks from the latest N entries, in reading order
    latest_chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            text         = doc,
            source       = meta.get("source", "unknown"),
            date         = meta.get("date", "unknown"),
            file_type    = meta.get("file_type", "note"),
            chunk_index  = int(meta.get("chunk_index", 0)),
            distance     = 0.0,
            hybrid_score = 1.0,
        )
        for _, doc, meta in zip(
            all_data["ids"], all_data["documents"], all_data["metadatas"]
        )
        if meta.get("source") in top_sources
    ]
    latest_chunks.sort(key=lambda c: (c.date, c.source, c.chunk_index))

    # Build a query from the latest entries' text to anchor the history search
    combined_text = " ".join(c.text for c in latest_chunks)
    query_words   = combined_text.split()
    query         = " ".join(query_words[:300])  # ~300 words is enough for embedding

    # Retrieve historically relevant chunks, excluding sources already covered
    history = _retriever.query(query, top_k=history_top_k, since=None)
    history_chunks = [c for c in history if c.source not in top_sources]

    return latest_chunks + history_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Quick CLI test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python retriever.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Querying: {question!r}\n")
    results  = retrieve(question)

    if not results:
        print("No results found. Is the vault indexed? Run: python indexer.py")
        sys.exit(0)

    for i, chunk in enumerate(results, 1):
        print(f"── Result {i} (hybrid={chunk.hybrid_score:.3f}, sem={chunk.score:.3f}) ──")
        print(f"   Source : {chunk.source}")
        print(f"   Date   : {chunk.date}")
        print(f"   Type   : {chunk.file_type}")
        print(f"   Text   : {chunk.text[:300]}…\n")
