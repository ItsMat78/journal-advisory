"""
main.py — CLI entrypoint for journal-rag.

Usage:
    journal "your question here"
    python main.py "your question here"

If the vault is not configured, setup.py is triggered automatically.
"""

import io
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from agents import run_multi_agent
from config import BASE_DIR, MODEL_DEFAULT, MODEL_DEEP, TEMP_DIR, TOP_K, get_vault_path
from prompt_builder import build_prompt
from retriever import retrieve, retrieve_latest
from ui import console, ok, err, warn


def _run_setup() -> None:
    """Invoke setup.py interactively."""
    setup_script = BASE_DIR / "setup.py"
    warn("Vault not configured. Launching first-time setup …")
    console.print()
    result = subprocess.run([sys.executable, str(setup_script)])
    if result.returncode != 0:
        console.print()
        err("Setup did not complete. Please run setup.py manually.")
        sys.exit(1)


def _check_claude_available() -> bool:
    """Return True if `claude` is available on PATH."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _print_help() -> None:
    from rich.panel import Panel

    def _section(title: str, sub: str = "") -> None:
        console.print()
        if sub:
            console.rule(f"[head]{title}[/head]  [muted]{sub}[/muted]", align="left", style="bright_black")
        else:
            console.rule(f"[head]{title}[/head]", align="left", style="bright_black")

    console.print()
    console.print(
        Panel(
            "[head]journal[/head]  —  personal journal analysis powered by local RAG + Claude\n"
            "[muted]Indexes your Obsidian vault locally, retrieves relevant entries,\n"
            "and runs them through Claude for self-reflection and pattern analysis.[/muted]",
            border_style="bright_black",
            padding=(0, 1),
        )
    )

    _section("ONE-SHOT MODE", "ask a question, get a response, done")
    console.print()
    console.print('  [cmd]journal[/cmd] [flag]"question"[/flag]')
    console.print()
    console.print('  [cmd]journal[/cmd] [flag]"What have I been stressed about lately?"[/flag]')
    console.print('  [cmd]journal[/cmd] [flag]"How was my relationship with Soumya this semester?"[/flag]')
    console.print(f'  [cmd]journal --latest[/cmd]                    [muted]go through the most recent entry[/muted]')
    console.print(f'  [cmd]journal --latest[/cmd] [flag]3[/flag]                  [muted]go through the 3 most recent entries[/muted]')

    _section("CHAT MODE", "multi-turn conversation, saved to disk")
    console.print()
    console.print(f'  [cmd]journal chat[/cmd]                        [muted]start a new session[/muted]')
    console.print(f'  [cmd]journal chat[/cmd] [flag]"opening question"[/flag]     [muted]start with a first message[/muted]')
    console.print(f'  [cmd]journal chat --resume[/cmd]               [muted]pick from recent sessions interactively[/muted]')
    console.print(f'  [cmd]journal chat --resume latest[/cmd]        [muted]auto-resume the most recent session[/muted]')
    console.print(f'  [cmd]journal chat --multi[/cmd]                [muted]every turn runs all three analysts + synthesis[/muted]')
    console.print()
    console.print('  [muted]Every turn retrieves fresh context. Full history is carried forward.[/muted]')
    console.print('  [muted]Sessions are saved as Markdown in ./sessions/.[/muted]')
    console.print('  [muted]In-session: /latest [N]  ·  /instruct [text]  ·  exit[/muted]')

    _section("MULTI-AGENT MODE", "therapist + advisor + critic → synthesis")
    console.print()
    console.print('  [cmd]journal --multi[/cmd] [flag]"question"[/flag]')
    console.print()
    console.print('  [muted]Runs three analysts in parallel on the same retrieved context:[/muted]')
    console.print('  [muted]  Therapist    — emotional patterns, psychological dynamics[/muted]')
    console.print('  [muted]  Life Advisor — goals, habits, decisions, leverage points[/muted]')
    console.print('  [muted]  Critic       — blind spots, self-deceptions, uncomfortable truths[/muted]')
    console.print('  [muted]Then a synthesiser weaves the three analyses into one response.[/muted]')
    console.print()
    console.print('  [cmd]journal --multi[/cmd] [flag]"What are my biggest patterns this year?"[/flag]')
    console.print(f'  [cmd]journal --multi --latest[/cmd]          [muted]three analysts on your most recent entry[/muted]')
    console.print('  [cmd]journal --multi --deep[/cmd] [flag]"Give me a full psychological profile"[/flag]')

    _section("OPTIONS", "work with all modes above")
    console.print()
    console.print(f'  [flag]--since[/flag] [cmd]YYYY-MM-DD[/cmd]   [muted]only search entries on or after this date[/muted]')
    console.print(f'                 [muted]journal --since 2026-04-01 "How was April?"[/muted]')
    console.print(f'                 [muted]journal chat --since 2026-01-01[/muted]')
    console.print(f'                 [muted]journal --multi --since 2026-05-01 "exam week"[/muted]')
    console.print()
    console.print(f'  [flag]--top-k[/flag] [cmd]N[/cmd]            [muted]chunks to retrieve per question (default: {TOP_K})[/muted]')
    console.print(f'                 [muted]increase for broad analysis questions[/muted]')
    console.print()
    console.print(f'  [flag]--deep[/flag]              [muted]stronger model for the final response[/muted]')
    console.print(f'                 [muted]one-shot / chat:  uses {MODEL_DEEP}[/muted]')
    console.print(f'                 [muted]multi-agent:      keeps agents on {MODEL_DEFAULT},[/muted]')
    console.print(f'                 [muted]                  upgrades synthesiser to {MODEL_DEEP}[/muted]')
    console.print(f'                 [muted]requires a Max subscription[/muted]')
    console.print()
    console.print(f'  [flag]--instruct[/flag] [cmd]"text"[/cmd]   [muted]extra directive sent to Claude (does not affect retrieval)[/muted]')
    console.print(f'                 [muted]journal "what\'s been hard?" --instruct "don\'t mention work"[/muted]')
    console.print(f'                 [muted]journal chat --instruct "focus on emotions, not logistics"[/muted]')
    console.print()
    console.print(f'  [flag]--dry-run[/flag]           [muted]print the assembled prompt without calling Claude[/muted]')
    console.print(f'                 [muted](one-shot mode only)[/muted]')

    _section("INDEXING")
    console.print()
    console.print('  [cmd]python indexer.py[/cmd]          [muted]re-index changed files only (fast)[/muted]')
    console.print('  [cmd]python indexer.py --force[/cmd]  [muted]re-embed everything from scratch[/muted]')
    console.print('  [cmd]python watcher.py[/cmd]          [muted]start background auto-indexer (runs on login)[/muted]')
    console.print()


def main() -> None:
    # ── Parse question ─────────────────────────────────────────────────────
    args = sys.argv[1:]

    if args and args[0] == "chat":
        from chat import main as chat_main
        chat_main(args[1:])
        sys.exit(0)

    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        sys.exit(0)

    # Optional flags
    dry_run  = False
    deep     = False
    multi    = False
    top_k    = TOP_K
    since:    str | None = None
    instruct: str | None = None
    latest_n: int | None = None   # None = not set; int = use retrieve_latest(n)
    question_parts: list[str] = []

    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            dry_run = True
        elif args[i] == "--deep":
            deep = True
        elif args[i] == "--multi":
            multi = True
        elif args[i] == "--instruct" and i + 1 < len(args):
            instruct = args[i + 1]
            i += 1
        elif args[i] == "--top-k" and i + 1 < len(args):
            try:
                top_k = int(args[i + 1])
                i += 1
            except ValueError:
                err(f"Invalid --top-k value: {args[i+1]}")
                sys.exit(1)
        elif args[i] == "--since" and i + 1 < len(args):
            raw = args[i + 1]
            try:
                datetime.strptime(raw, "%Y-%m-%d")
                since = raw
            except ValueError:
                err(f"Invalid --since date: {raw!r}  (use YYYY-MM-DD)")
                sys.exit(1)
            i += 1
        elif args[i] == "--latest":
            # Optional numeric argument: --latest 3
            if i + 1 < len(args) and args[i + 1].isdigit():
                latest_n = int(args[i + 1])
                i += 1
            else:
                latest_n = 1
        else:
            question_parts.append(args[i])
        i += 1

    # --latest doesn't require a question; a default is supplied if omitted
    if latest_n is not None:
        if not question_parts:
            n_label  = f"{latest_n} " if latest_n > 1 else ""
            question = f"Go through {'these ' if latest_n > 1 else 'this '}journal {n_label}{'entries' if latest_n > 1 else 'entry'} and share what stands out."
        else:
            question = " ".join(question_parts)
    else:
        if not question_parts:
            err("No question provided.")
            sys.exit(1)
        question = " ".join(question_parts)

    model = MODEL_DEEP if deep else MODEL_DEFAULT

    # ── Ensure vault is configured ─────────────────────────────────────────
    if not get_vault_path():
        _run_setup()
        if not get_vault_path():
            print("Setup incomplete. Exiting.", file=sys.stderr)
            sys.exit(1)

    # ── Retrieve relevant chunks ───────────────────────────────────────────
    if latest_n is not None:
        n_label  = f"{latest_n} " if latest_n > 1 else ""
        msg      = f"Fetching {n_label}most recent {'entries' if latest_n > 1 else 'entry'} …"
        with console.status(f"[info]{msg}[/info]", spinner="dots"):
            chunks = retrieve_latest(n_entries=latest_n)
    else:
        since_label = f" [muted](since {since})[/muted]" if since else ""
        with console.status(f"[info]Searching journal …[/info]", spinner="dots"):
            chunks = retrieve(question, top_k=top_k, since=since)

    if not chunks:
        console.print()
        err("No relevant journal entries found.")
        console.print("  [muted]• The vault hasn't been indexed yet — run: python indexer.py[/muted]", stderr=True)
        console.print("  [muted]• The query doesn't match any existing entries.[/muted]", stderr=True)
        sys.exit(1)

    ok(f"Retrieved {len(chunks)} chunk(s).")

    if not _check_claude_available():
        console.print()
        err("`claude` command not found on PATH.")
        console.print("  [muted]Make sure Claude Code CLI is installed: https://claude.ai/code[/muted]", stderr=True)
        sys.exit(1)

    # ── Multi-agent path ───────────────────────────────────────────────────
    if multi:
        if dry_run:
            err("--dry-run is not supported with --multi.")
            sys.exit(1)
        exit_code = run_multi_agent(
            question         = question,
            chunks           = chunks,
            model            = MODEL_DEFAULT,
            synthesis_model  = model,   # MODEL_DEEP if --deep, else MODEL_DEFAULT
            instruct         = instruct,
        )
        sys.exit(exit_code)

    # ── Single-agent path ──────────────────────────────────────────────────
    prompt = build_prompt(question, chunks, instruct=instruct)

    if dry_run:
        console.print()
        console.rule("[head]DRY RUN — prompt that would be sent to Claude[/head]", style="bright_black")
        console.print()
        console.print(prompt)
        sys.exit(0)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".txt", prefix="journal-rag-", dir=str(TEMP_DIR)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(prompt)

        if instruct:
            console.print(f"  [muted]Instruction:[/muted] [flag]{instruct}[/flag]")
        console.print(f"\n  [head]Launching Claude[/head] [muted][{model}][/muted] …\n")
        with open(tmp_path, "r", encoding="utf-8") as prompt_file:
            result = subprocess.run(
                ["claude", "--print", "--model", model],
                stdin=prompt_file,
                text=True,
            )
        sys.exit(result.returncode)

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
