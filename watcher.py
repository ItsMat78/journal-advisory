"""
watcher.py — Monitors the Obsidian vault for file changes and re-indexes
modified or newly created markdown files automatically.

This script is designed to run continuously in the background (registered
as a Windows Task Scheduler job by setup.py).

Log output goes to journal-rag.log in the project directory.
"""

import logging
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config import LOG_FILE, SKIP_DIRS, SUPPORTED_EXTENSIONS, get_vault_path
from indexer import (
    get_chroma_client,
    get_collection,
    index_file,
    load_embedding_model,
    remove_file_from_index,
)

# Logging is configured by indexer's module-level basicConfig (imported above).
logger = logging.getLogger(__name__)


def _should_skip(filepath: Path, vault_path: Path) -> bool:
    """Return True if this file should never be indexed."""
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return True
    try:
        rel = filepath.relative_to(vault_path)
    except ValueError:
        return True
    for part in rel.parts:
        if part in SKIP_DIRS or part.lower() == "images" or part.startswith("."):
            return True
    return False


class VaultEventHandler(FileSystemEventHandler):
    _DEBOUNCE_DELAY = 1.5  # seconds — absorbs rapid save bursts from editors

    def __init__(self, vault_path: Path, collection, model):
        super().__init__()
        self.vault_path = vault_path
        self.collection = collection
        self.model      = model
        self._pending: dict[str, threading.Timer] = {}
        self._lock      = threading.Lock()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _schedule_upsert(self, path_str: str) -> None:
        """Cancel any pending re-index for this path and schedule a fresh one."""
        with self._lock:
            existing = self._pending.pop(path_str, None)
            if existing:
                existing.cancel()
            t = threading.Timer(self._DEBOUNCE_DELAY, self._handle_upsert, args=(path_str,))
            self._pending[path_str] = t
            t.start()

    def _handle_upsert(self, path_str: str) -> None:
        with self._lock:
            self._pending.pop(path_str, None)
        fp = Path(path_str)
        if _should_skip(fp, self.vault_path):
            return
        # index_file handles mtime check, stale-chunk removal, and re-indexing
        # in one call. It returns -1 if unchanged, 0 if empty, N if indexed.
        n = index_file(fp, self.vault_path, self.collection, self.model)
        if n > 0:
            logger.info(f"Re-indexed '{fp.name}': {n} chunks")
        elif n == 0:
            logger.debug(f"No indexable content in '{fp.name}'")
        # n == -1 means mtime unchanged (rare in watcher, but safe to ignore)

    def _handle_delete(self, path_str: str) -> None:
        fp = Path(path_str)
        if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        logger.info(f"Deletion detected: {fp.name} — removing from index ...")
        remove_file_from_index(fp, self.vault_path, self.collection)

    def _handle_move(self, src_str: str, dst_str: str) -> None:
        src = Path(src_str)
        dst = Path(dst_str)
        if src.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info(f"Move: {src.name} -> {dst.name}")
            remove_file_from_index(src, self.vault_path, self.collection)
        if not _should_skip(dst, self.vault_path):
            n = index_file(dst, self.vault_path, self.collection, self.model, force=True)
            if n > 0:
                logger.info(f"Indexed moved file '{dst.name}': {n} chunks")

    # ── watchdog callbacks ────────────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_upsert(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_upsert(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_delete(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_move(event.src_path, event.dest_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    vault_path_str = get_vault_path()
    if not vault_path_str:
        logger.error("Vault path not configured. Run setup.py first.")
        sys.exit(1)

    vault_path = Path(vault_path_str)
    if not vault_path.is_dir():
        logger.error(f"Vault directory not found: {vault_path}")
        sys.exit(1)

    logger.info("Loading embedding model ...")
    model = load_embedding_model()

    client     = get_chroma_client()
    collection = get_collection(client)

    handler  = VaultEventHandler(vault_path, collection, model)
    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)
    observer.start()

    logger.info(f"Watcher started — monitoring: {vault_path}")

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Watcher stopped by user.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
