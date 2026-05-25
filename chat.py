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
from retriever import retrieve, retrieve_current_month, retrieve_latest, retrieve_stratified
from rich.markup import escape
from rich.text import Text
from ui import console, warn

SESSIONS_DIR = BASE_DIR / "sessions"

_PERSONA = """\
You are a reflective journal analysis assistant having an ongoing conversation \
with the journal's author.

How you operate:
- You have complete visibility of all journal entries from the current calendar month \
(loaded automatically — treat this as your working memory of recent life)
- Each message also includes historically relevant excerpts and people notes retrieved \
by semantic search
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

      "    ___       __      _                       \n"
      "   /   | ____/ /   __(_)________  _______  __ \n"
      "  / /| |/ __  / | / / / ___/ __ \/ ___/ / / / \n"
      " / ___ / /_/ /| |/ / (__  ) /_/ / /  / /_/ /  \n"
      "/_/  |_\__,_/ |___/_/____/\____/_/   \__, /   \n"
      "                                    /____/     "
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
    path: Path, turn: int, question: str,
    short_term_chunks, long_term_chunks, response: str
) -> None:
    lt_sources = sorted(set(f"`{c.source}` ({c.date})" for c in long_term_chunks))
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"\n## Turn {turn}\n\n"
            f"**You:** {question}\n\n"
            f"**Month context:** {len(short_term_chunks)} chunks  \n"
            f"**Retrieved:** {', '.join(lt_sources) or 'none'}\n\n"
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


def _prompt_rename(path: Path) -> Path:
    """Ask the user for an optional name suffix and rename the session file."""
    try:
        name = console.input("\n  [dim]Name this session (Enter to skip):[/dim] ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    if not name:
        return path
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    if not slug:
        return path
    new_path = path.with_name(f"{path.stem}_{slug}.md")
    path.rename(new_path)
    return new_path


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
    short_term_chunks,
    long_term_chunks,
    history: list[dict],
    instruct: str | None = None,
) -> str:
    parts = [_PERSONA]

    if history:
        conv = "\n\n---\n\n".join(
            f"User: {t['question']}\n\nAssistant: {t['response']}"
            for t in history
        )
        parts.append(f"\n{_D}\nCONVERSATION HISTORY\n{_D}\n\n{conv}")

    if short_term_chunks:
        ctx_st = format_context_block(short_term_chunks)
        parts.append(
            f"\n{_D}\nCURRENT MONTH — COMPLETE JOURNAL RECORD\n"
            f"({len(short_term_chunks)} chunks · loaded automatically · chronological order)\n"
            f"{_D}\n\n{ctx_st}"
        )

    if long_term_chunks:
        ctx_lt = format_context_block(long_term_chunks)
        parts.append(
            f"\n{_D}\nHISTORICAL & PEOPLE CONTEXT — RETRIEVED FOR THIS MESSAGE\n"
            f"({len(long_term_chunks)} chunks · semantic search · older entries + people notes)\n"
            f"{_D}\n\n{ctx_lt}"
        )

    if not short_term_chunks and not long_term_chunks:
        parts.append(f"\n{_D}\nJOURNAL CONTEXT\n{_D}\n\n(No relevant entries found.)")

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
    until: str | None,
    instruct: str | None,
    month_chunks: int = 0,
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

    t.append("   this month  ", style="dim")
    if month_chunks:
        t.append(f"{month_chunks} chunks loaded\n", style="cyan")
    else:
        t.append("no entries yet\n", style="dim")

    if since or until:
        t.append("   range     ", style="dim")
        label = since or ""
        if until:
            label = f"{label} → {until}" if since else f"until {until}"
        t.append(f"{label}\n", style="cyan")
    if instruct:
        t.append("   instruct  ", style="dim")
        t.append(f"{instruct}\n", style="cyan")

    t.append(f"\n   {DIV}\n\n", style="bright_black")

    t.append("   /multi        ", style="bold cyan")
    t.append("toggle multi-agent mode\n", style="dim")
    t.append("   /latest [N] [q]", style="bold cyan")
    t.append("pull N most recent entries, optional question\n", style="dim")
    t.append("   /since [d] [d] ", style="bold cyan")
    t.append("focus on date or range (YYYY-MM-DD), clear with /since\n", style="dim")
    t.append("   /top-k N      ", style="bold cyan")
    t.append("change chunks retrieved per turn\n", style="dim")
    t.append("   /instruct     ", style="bold cyan")
    t.append("set response directive  ", style="dim")
    t.append("/instruct alone to clear\n", style="dim italic")
    t.append("   /refresh      ", style="bold cyan")
    t.append("re-index changed vault files now\n", style="dim")
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
    until: str | None,
    instruct: str | None,
    month_chunks: int = 0,
) -> None:
    parts: list[str] = []
    if multi:
        parts.append("[bold cyan]◈ multi[/bold cyan]")
    else:
        parts.append("[dim]◆ single[/dim]")
    parts.append(f"[dim]{_short_model(model)}[/dim]")
    parts.append(f"[dim]k={top_k}[/dim]")
    if month_chunks:
        parts.append(f"[dim]month:{month_chunks}[/dim]")
    if since or until:
        label = since or ""
        if until:
            label = f"{label}→{until}" if since else f"until {until}"
        parts.append(f"[cyan]{label}[/cyan]")
    if instruct:
        s = (instruct[:25] + "…") if len(instruct) > 25 else instruct
        s = s.replace("[", "\\[")
        parts.append(f'[dim cyan]"{s}"[/dim cyan]')

    settings = "  ·  ".join(parts)
    commands  = "[dim]/multi  /latest  /since  /top-k  /instruct  /refresh  exit[/dim]"
    console.rule(f"{settings}    {commands}", style="bright_black")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────────────────────────────────────────

def main(cli_args: list[str]) -> None:
    """Entry point called from main.py when `journal chat [...]` is used."""

    model     = MODEL_DEFAULT
    top_k     = TOP_K
    since     = None
    until: str | None = None
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

    with console.status("[info]Loading current month …[/info]", spinner="dots"):
        current_month_chunks = retrieve_current_month()

    flag_parts = []
    if since:             flag_parts.append(f"--since {since}")
    if top_k != TOP_K:   flag_parts.append(f"--top-k {top_k}")
    if model != MODEL_DEFAULT: flag_parts.append("--deep")
    if multi:             flag_parts.append("--multi")
    if instruct:          flag_parts.append(f'--instruct "{instruct}"')

    if is_new:
        _init_session(session_path, model, " ".join(flag_parts))

    _print_welcome(session_path, is_new, model, multi, top_k, since, until, instruct,
                   month_chunks=len(current_month_chunks))

    turn_num = len(history)
    pending  = initial_q

    while True:
        # ── Statusline + prompt ───────────────────────────────────────────
        _statusline(model, multi, top_k, since, until, instruct, month_chunks=len(current_month_chunks))

        _from_input = False
        if pending:
            question = pending
            pending  = None
        else:
            try:
                question     = console.input("[bold cyan]>[/bold cyan] ").strip()
                _from_input  = True
            except (KeyboardInterrupt, EOFError):
                session_path = _prompt_rename(session_path)
                console.print(f"\n  [muted]Session saved → {session_path.name}[/muted]")
                break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "/exit"):
            session_path = _prompt_rename(session_path)
            console.print(f"\n  [muted]Session saved → {session_path.name}[/muted]")
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

        # ── /since command ─────────────────────────────────────────────────
        if question.lower().startswith("/since"):
            _ISO = r"^\d{4}-\d{2}-\d{2}$"
            import re as _re
            tokens = question.split()
            dates  = [t for t in tokens[1:] if _re.match(_ISO, t)]
            if not tokens[1:]:
                since = None
                until = None
                console.print("\n  [muted]Date filter cleared. Back to normal mode.[/muted]\n")
            elif len(dates) == 0:
                console.print(
                    "\n  [warn]⚠[/warn]  Usage: [cmd]/since YYYY-MM-DD [YYYY-MM-DD][/cmd]\n"
                )
            else:
                since = dates[0]
                until = dates[1] if len(dates) > 1 else None
                label = f"{since} → {until}" if until else since
                console.print(f"\n  [ok]✓[/ok]  Date filter → [cyan]{label}[/cyan]\n")
            continue

        # ── /refresh command ───────────────────────────────────────────────
        if question.lower() == "/refresh":
            console.print()
            console.print("  [info]Re-indexing vault (changed files only) …[/info]")
            console.print()
            from indexer import index_vault
            index_vault()
            with console.status("[info]Refreshing current month …[/info]", spinner="dots"):
                current_month_chunks = retrieve_current_month()
            console.print(f"  [ok]✓[/ok]  Current month: [cyan]{len(current_month_chunks)}[/cyan] chunks loaded")
            console.print()
            continue

        # ── /latest command ────────────────────────────────────────────────
        use_latest        = None
        actual_q          = question
        latest_instruct   = None

        if question.lower().startswith("/latest"):
            tokens     = question.split(maxsplit=2)
            # /latest [N] [question...]
            n_arg      = tokens[1] if len(tokens) > 1 else ""
            use_latest = int(n_arg) if n_arg.isdigit() else 1
            user_q     = tokens[2] if n_arg.isdigit() and len(tokens) > 2 else (
                         tokens[1] if not n_arg.isdigit() and len(tokens) > 1 else ""
            )
            actual_q   = user_q or (
                f"Go through {'these' if use_latest > 1 else 'this'} "
                f"journal {'entries' if use_latest > 1 else 'entry'} and share what stands out."
            )
            latest_instruct = (
                "Ground your answer in the recent entries provided above — that is the primary focus. "
                "Historical context is background knowledge; only reference it if it directly "
                "illuminates the user's current situation. Be concrete, present-tense, and specific. "
                "Do not pivot to long-term pattern analysis unless the user explicitly asks."
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
                # /latest: bypass short-term, use full retrieve_latest behaviour
                st_chunks = []
                lt_chunks = retrieve_latest(n_entries=use_latest)
            elif since or until:
                # date filter: stratified retrieval for multi-month ranges
                st_chunks = []
                lt_chunks = retrieve_stratified(actual_q, top_k=top_k, since=since, until=until)
            else:
                # normal turn: current month pinned + long-term RAG
                st_chunks = current_month_chunks
                lt_chunks = retrieve(actual_q, top_k=top_k, long_term_only=True)

        all_chunks = st_chunks + lt_chunks
        if not all_chunks:
            warn("No relevant entries found.")
            console.print()
            continue

        if st_chunks and lt_chunks:
            console.print(
                f"  [muted]Month: {len(st_chunks)} chunks  ·  Retrieved: {len(lt_chunks)} chunk(s).[/muted]"
            )
        elif lt_chunks:
            console.print(f"  [muted]Retrieved {len(lt_chunks)} chunk(s).[/muted]")
        else:
            console.print(f"  [muted]Month: {len(st_chunks)} chunks (no historical matches).[/muted]")
        console.print()

        # ── Call Claude ────────────────────────────────────────────────────
        # Merge /latest focus instruction with any active /instruct
        effective_instruct = instruct
        if latest_instruct:
            effective_instruct = (
                f"{latest_instruct}\n\n{instruct}" if instruct else latest_instruct
            )

        if multi:
            result   = run_multi_agent(
                question        = actual_q,
                chunks          = all_chunks,
                model           = model,
                synthesis_model = model,
                instruct        = effective_instruct,
                capture_output  = True,
            )
            response = result if isinstance(result, str) else ""
        else:
            prompt   = _build_prompt(actual_q, st_chunks, lt_chunks, history, instruct=effective_instruct)
            response = _call_streaming(prompt, model)

        # ── Persist ────────────────────────────────────────────────────────
        turn_num += 1
        history.append({"question": actual_q, "response": response})
        _append_turn(session_path, turn_num, actual_q, st_chunks, lt_chunks, response)
