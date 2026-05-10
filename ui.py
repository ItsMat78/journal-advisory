"""
ui.py — Shared Rich console and formatting helpers for journal-rag.
"""
import io
import sys

from rich.console import Console
from rich.theme import Theme

# Rich on Windows may fall back to the legacy Win32 console API (cp1252),
# which cannot encode Unicode box-drawing characters.  Passing an explicit
# UTF-8 TextIOWrapper bypasses that path entirely.
_stdout = (
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
    if hasattr(sys.stdout, "buffer")
    else sys.stdout
)

console = Console(
    file=_stdout,
    theme=Theme({
        "ok":    "bold green",
        "warn":  "bold yellow",
        "err":   "bold red",
        "info":  "dim cyan",
        "cmd":   "bold cyan",
        "flag":  "cyan",
        "head":  "bold",
        "muted": "dim",
    }),
    highlight=False,
    force_terminal=True,    # Always emit ANSI codes (Windows Terminal supports VT)
    legacy_windows=False,   # Never fall back to legacy Win32 console API
)


def ok(msg: str) -> None:
    console.print(f"[ok]✓[/ok]  {msg}")


def warn(msg: str) -> None:
    console.print(f"[warn]⚠[/warn]  {msg}")


def err(msg: str) -> None:
    console.print(f"[err]✗[/err]  {msg}", stderr=True)
