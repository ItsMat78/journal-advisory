"""
chat.py — Interactive multi-turn chat with per-turn retrieval.

Each turn: query rewrite → retrieve fresh chunks → call Claude with
full conversation history → stream response → append to session file.

Sessions are saved as Markdown in ./sessions/.
"""

import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from agents import run_multi_agent
from config import BASE_DIR, MODEL_DEFAULT, MODEL_DEEP, TOP_K, get_vault_path
from prompt_builder import format_context_block
from retriever import retrieve, retrieve_latest
from rich.markup import escape
from rich.text import Text
from ui import console, warn

SESSIONS_DIR = BASE_DIR / "sessions"

_PERSONA = """\
You are a reflective journal analysis assistant having an ongoing conversation \
with the journal's author.

How you operate:
- Each message comes with freshly retrieved journal excerpts relevant to the question
- You have the full conversation history above — build on it, don't repeat yourself
- Be analytical and direct. Name patterns plainly, even uncomfortable ones
- Ground every observation in the actual text. Quote or paraphrase specific entries
- If the context is insufficient, say so rather than speculating
- Treat all journal content with discretion\
""".strip()

_D = "=" * 72

_LOGO = (
    "     ██╗ ██████╗ ██╗   ██╗██████╗ ███╗   ██╗ █████╗ ██╗     \n"
    "     ██║██╔═══██╗██║   ██║██╔══██╗████╗  ██║██╔══██╗██║     \n"
    "     ██║██║   ██║██║   ██║██████╔╝██╔██╗ ██║███████║██║     \n"
    "██   ██║██║   ██║██║   ██║██╔══██╗██║╚██╗██║██╔══██║██║     \n"
    "╚█████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║██║  ██║███████╗\n"
    " ╚════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝"
)

_LOGO_SUB = (
    "    ___       __      _                      \n"
    "   /   | ____/ /   __(_)________  _______  __\n"
    "  / /| |/ __  / | / / / ___/ __ \\/ ___/ / / /\n"
    " / ___ / /_/ /| |/ / (__  ) /_/ / /  / /_/ / \n"
    "/_/  |_\\__,_/ |___/_/____/\\____/_/   \\__, /  \n"
    "                                      /____/  "
)


# ─────────────────────────────────────────────────────────────────────────────
# Session file I/O
# ─────────────────────────────────────────────────────────────────────────────

def _init_session(path: Path, model: str, flags: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(
        f"# Journal Chat — {now}\n\n"
        f"**Model:** `{model}`  \n"
        f"**Flags:** {flags or 'none'}\n\n"
        "---\n",
        encoding="utf-8",
    )


def _append_turn(
    path: Path, turn: int, question: str, chunks, response: str
) -> None:
    sources = sorted(set(f"`{c.source}` ({c.date})" for c in chunks))
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"\n## Turn {turn}\n\n"
            f"**You:** {question}\n\n"
            f"**Retrieved:** {', '.join(sources) or 'none'}\n\n"
            f"**Claude:**\n\n{response}\n\n"
            "---\n"
        )


def _parse_session(path: Path) -> list[dict]:
    """Extract conversation history from a session MD file."""
    text = path.read_text(encoding="utf-8")
    turns = []
    for section in re.split(r"\n## Turn \d+\n", text)[1:]:
        q = re.search(r"\*\*You:\*\* (.+?)\n\n", section, re.DOTALL)
        r = re.search(r"\*\*Claude:\*\*\n\n(.+?)(?=\n\n---|$)", section, re.DOTALL)
        if q and r:
            turns.append({
                "question": q.group(1).strip(),
                "response": r.group(1).strip(),
            })
    return turns


def _count_turns(path: Path) -> int:
    return len(re.findall(r"\n## Turn \d+", path.read_text(encoding="utf-8")))


# ─────────────────────────────────────────────────────────────────────────────
# Session selection
# ─────────────────────────────────────────────────────────────────────────────

def _pick_session(resume: str | None) -> tuple[Path, list[dict]]:
    """
    Returns (session_path, prior_history).
    resume=None   → new session
    resume=""     → interactive picker
    resume="latest" → auto-select most recent
    resume="<stem>" → specific session by filename stem
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SESSIONS_DIR.glob("*.md"), reverse=True)

    def _new() -> tuple[Path, list[dict]]:
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return SESSIONS_DIR / f"{name}.md", []

    if resume is None:
        return _new()

    if not existing:
        console.print("[muted]No previous sessions found. Starting a new one.[/muted]\n")
        return _new()

    if resume == "latest":
        path = existing[0]
        console.print(f"[muted]Resuming:[/muted] [cmd]{path.name}[/cmd]\n")
        return path, _parse_session(path)

    if resume and resume != "":
        # Match by stem or partial name
        matches = [p for p in existing if resume in p.stem]
        if matches:
            path = matches[0]
            console.print(f"[muted]Resuming:[/muted] [cmd]{path.name}[/cmd]\n")
            return path, _parse_session(path)
        console.print(f"[warn]Session '{resume}' not found. Starting a new one.[/warn]\n")
        return _new()

    # Interactive picker
    show = existing[:10]
    console.print("[head]Recent sessions[/head]")
    for i, p in enumerate(show, 1):
        turns = _count_turns(p)
        ts    = p.stem.replace("_", " ", 1).replace("-", "/", 2)
        console.print(f"  [muted]{i}.[/muted] [cmd]{ts}[/cmd] [muted]— {turns} turn{'s' if turns != 1 else ''}[/muted]")
    console.print()
    try:
        raw = console.input("[muted]Resume which? (number, or Enter for new):[/muted] ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(show):
            path = show[idx]
            console.print(f"\n[muted]Resuming:[/muted] [cmd]{path.name}[/cmd]\n")
            return path, _parse_session(path)
    return _new()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(
    question: str,
    chunks,
    history: list[dict],
    instruct: str | None = None,
) -> str:
    ctx   = format_context_block(chunks)
    parts = [_PERSONA]

    if history:
        conv = "\n\n---\n\n".join(
            f"User: {t['question']}\n\nAssistant: {t['response']}"
            for t in history
        )
        parts.append(f"\n{_D}\nCONVERSATION HISTORY\n{_D}\n\n{conv}")

    parts.append(
        f"\n{_D}\nRELEVANT JOURNAL EXCERPTS (fresh for this message)\n{_D}\n\n{ctx}"
    )
    parts.append(f"\n{_D}\nUSER\n{_D}\n\n{question}")
    if instruct:
        parts.append(f"\n{_D}\nADDITIONAL INSTRUCTIONS\n{_D}\n\n{instruct}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Claude call — streams to stdout, returns full text
# ─────────────────────────────────────────────────────────────────────────────

def _call_streaming(prompt: str, model: str) -> str:
    tmp_dir = BASE_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".txt", prefix="journal-chat-", dir=str(tmp_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(prompt)

        stdin_f = open(tmp_path, "r", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                ["claude", "--print", "--model", model],
                stdin=stdin_f,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
            )
            out = []
            while True:
                block = proc.stdout.read(64)
                if not block:
                    break
                print(block, end="", flush=True)
                out.append(block)
            proc.wait()
            print("\n")
            return "".join(out).strip()
        finally:
            stdin_f.close()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI — welcome screen + per-turn statusline
# ─────────────────────────────────────────────────────────────────────────────

def _short_model(model: str) -> str:
    return (
        model.replace("claude-", "")
             .replace("-4-6", "")
             .replace("-4-5-20251001", "")
    )


def _print_welcome(
    session_path: Path,
    is_new: bool,
    model: str,
    multi: bool,
    top_k: int,
    since: str | None,
    instruct: str | None,
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    DIV = "─" * 58

    t = Text()
    t.append("\n")

    # ASCII art logo
    for line in _LOGO.split("\n"):
        t.append(f"  {line}\n", style="bold cyan")
    for line in _LOGO_SUB.split("\n"):
        t.append(f"        {line}\n", style="dim cyan")
    t.append("\n")

    t.append("   chat mode  ·  personal journal analysis  ·  local RAG + Claude\n", style="dim")
    t.append(f"\n   {DIV}\n\n", style="bright_black")

    # Session settings
    t.append("   session   ", style="dim")
    t.append(session_path.name)
    t.append(f"   [{'new' if is_new else 'resumed'}]\n",
             style="green bold" if is_new else "cyan bold")

    t.append("   model     ", style="dim")
    t.append(f"{_short_model(model)}\n")

    t.append("   mode      ", style="dim")
    if multi:
        t.append("◈ multi-agent\n", style="bold cyan")
    else:
        t.append("◆ single-agent\n")

    t.append("   top-k     ", style="dim")
    t.append(f"{top_k}\n")

    if since:
        t.append("   since     ", style="dim")
        t.append(f"{since}\n", style="cyan")
    if instruct:
        t.append("   instruct  ", style="dim")
        t.append(f"{instruct}\n", style="cyan")

    t.append(f"\n   {DIV}\n\n", style="bright_black")

    t.append("   /multi        ", style="bold cyan")
    t.append("toggle multi-agent mode\n", style="dim")
    t.append("   /latest [N]   ", style="bold cyan")
    t.append("pull N most recent entries\n", style="dim")
    t.append("   /top-k N      ", style="bold cyan")
    t.append("change chunks retrieved per turn\n", style="dim")
    t.append("   /instruct     ", style="bold cyan")
    t.append("set response directive  ", style="dim")
    t.append("/instruct alone to clear\n", style="dim italic")
    t.append("   exit          ", style="bold cyan")
    t.append("save session and quit\n", style="dim")
    t.append("\n")

    console.print(Panel(t, border_style="bright_black", padding=(0, 0)))
    console.print()


def _statusline(
    model: str,
    multi: bool,
    top_k: int,
    since: str | None,
    instruct: str | None,
) -> None:
    parts: list[str] = []
    if multi:
        parts.append("[bold cyan]◈ multi[/bold cyan]")
    else:
        parts.append("[dim]◆ single[/dim]")
    parts.append(f"[dim]{_short_model(model)}[/dim]")
    parts.append(f"[dim]k={top_k}[/dim]")
    if since:
        parts.append(f"[cyan]since {since}[/cyan]")
    if instruct:
        s = (instruct[:25] + "…") if len(instruct) > 25 else instruct
        s = s.replace("[", "\\[")
        parts.append(f'[dim cyan]"{s}"[/dim cyan]')

    settings = "  ·  ".join(parts)
    commands  = "[dim]/multi  /latest  /top-k  /instruct  exit[/dim]"
    console.rule(f"{settings}    {commands}", style="bright_black")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────────────────────────────────────────

def main(cli_args: list[str]) -> None:
    """Entry point called from main.py when `journal chat [...]` is used."""

    model     = MODEL_DEFAULT
    top_k     = TOP_K
    since     = None
    instruct: str | None = None
    multi     = False
    resume    = None   # None=new, ""=picker, "latest"=auto, "<stem>"=specific
    initial_q = None

    i = 0
    while i < len(cli_args):
        a = cli_args[i]
        if a == "--deep":
            model = MODEL_DEEP
        elif a == "--multi":
            multi = True
        elif a == "--top-k" and i + 1 < len(cli_args):
            try:
                top_k = int(cli_args[i + 1])
                i += 1
            except ValueError:
                pass
        elif a == "--since" and i + 1 < len(cli_args):
            since = cli_args[i + 1]
            i += 1
        elif a == "--instruct" and i + 1 < len(cli_args):
            instruct = cli_args[i + 1]
            i += 1
        elif a == "--resume":
            if i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
                resume = cli_args[i + 1]
                i += 1
            else:
                resume = ""   # trigger interactive picker
        elif not a.startswith("--"):
            initial_q = a
        i += 1

    if not get_vault_path():
        from ui import err
        err("Vault not configured. Run setup.py first.")
        sys.exit(1)

    session_path, history = _pick_session(resume)
    is_new = not history

    flag_parts = []
    if since:             flag_parts.append(f"--since {since}")
    if top_k != TOP_K:   flag_parts.append(f"--top-k {top_k}")
    if model != MODEL_DEFAULT: flag_parts.append("--deep")
    if multi:             flag_parts.append("--multi")
    if instruct:          flag_parts.append(f'--instruct "{instruct}"')

    if is_new:
        _init_session(session_path, model, " ".join(flag_parts))

    _print_welcome(session_path, is_new, model, multi, top_k, since, instruct)

    turn_num = len(history)
    pending  = initial_q

    while True:
        # ── Statusline + prompt ───────────────────────────────────────────
        _statusline(model, multi, top_k, since, instruct)

        _from_input = False
        if pending:
            question = pending
            pending  = None
        else:
            try:
                question     = console.input("[bold cyan]>[/bold cyan] ").strip()
                _from_input  = True
            except (KeyboardInterrupt, EOFError):
                console.print(f"\n  [muted]Session saved → {session_path}[/muted]")
                break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "/exit"):
            console.print(f"\n  [muted]Session saved → {session_path}[/muted]")
            break

        # ── /top-k command ────────────────────────────────────────────────
        if question.lower().startswith("/top-k") or question.lower().startswith("/k "):
            prefix = "/top-k" if question.lower().startswith("/top-k") else "/k"
            rest   = question[len(prefix):].strip()
            if rest.isdigit() and int(rest) > 0:
                top_k = int(rest)
                console.print(f"\n  [ok]✓[/ok]  top-k → [cyan]{top_k}[/cyan] chunks per turn\n")
            else:
                console.print(f"\n  [warn]⚠[/warn]  Usage: [cmd]/top-k N[/cmd]  (N must be a positive integer)\n")
            continue

        # ── /multi toggle ──────────────────────────────────────────────────
        if question.lower() == "/multi":
            multi = not multi
            if multi:
                console.print("\n  [ok]✓[/ok]  Mode → [bold cyan]◈ multi-agent[/bold cyan]  [dim](therapist · advisor · critic → synthesis)[/dim]\n")
            else:
                console.print("\n  [ok]✓[/ok]  Mode → [dim]◆ single-agent[/dim]\n")
            continue

        # ── /instruct command ──────────────────────────────────────────────
        if question.lower().startswith("/instruct"):
            rest = question[len("/instruct"):].strip()
            if rest:
                instruct = rest
                console.print(f"\n  [ok]✓[/ok]  Instruction → [cyan]{instruct}[/cyan]\n")
            else:
                instruct = None
                console.print("\n  [muted]Instruction cleared.[/muted]\n")
            continue

        # ── /latest command ────────────────────────────────────────────────
        use_latest  = None
        actual_q    = question

        if question.lower().startswith("/latest"):
            parts      = question.split(maxsplit=1)
            use_latest = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            actual_q   = (
                f"Go through {'these' if use_latest > 1 else 'this'} "
                f"journal {'entries' if use_latest > 1 else 'entry'} and share what stands out."
            )

        # ── User input — grey background (Claude Code style) ──────────────
        if _from_input:
            # Erase the terminal-echoed "> typed text" line(s) before reprinting styled.
            # Prompt is 2 visible chars ("> "); calculate how many lines the input occupied.
            lines_used = max(1, math.ceil((2 + len(question)) / (console.width or 80)))
            for _ in range(lines_used):
                console.file.write("\x1b[1A\x1b[2K")   # cursor up, erase line
            console.file.flush()
        else:
            console.print()   # blank line before pending/pre-set question

        _msg = Text()
        _msg.append("  ")
        _msg.append(question)
        console.print(_msg, style="on color(240)")

        # ── Retrieve ───────────────────────────────────────────────────────
        with console.status("[info]Retrieving …[/info]", spinner="dots"):
            if use_latest is not None:
                chunks = retrieve_latest(n_entries=use_latest)
            else:
                chunks = retrieve(actual_q, top_k=top_k, since=since)

        if not chunks:
            warn("No relevant entries found.")
            console.print()
            continue

        console.print(f"  [muted]Retrieved {len(chunks)} chunk(s).[/muted]")
        console.print()

        # ── Call Claude ────────────────────────────────────────────────────
        if multi:
            result   = run_multi_agent(
                question        = actual_q,
                chunks          = chunks,
                model           = model,
                synthesis_model = model,
                instruct        = instruct,
                capture_output  = True,
            )
            response = result if isinstance(result, str) else ""
        else:
            prompt   = _build_prompt(actual_q, chunks, history, instruct=instruct)
            response = _call_streaming(prompt, model)

        # ── Persist ────────────────────────────────────────────────────────
        turn_num += 1
        history.append({"question": actual_q, "response": response})
        _append_turn(session_path, turn_num, actual_q, chunks, response)
