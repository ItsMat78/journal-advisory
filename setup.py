"""
setup.py — Interactive first-run configuration for journal-rag.

What this script does:
  1. Asks for the Obsidian vault path and saves it to config.json
  2. Installs required Python packages (pip install -r requirements.txt)
  3. Performs an initial full index of the vault
  4. Registers watcher.py to start on login (Task Scheduler, with Startup folder fallback)

Run:
    python setup.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ── Bootstrap: make sure config is importable even before deps are installed
BASE_DIR = Path(__file__).parent.resolve()

# ── Rich is optional here — it may not be installed yet on first run ──────────
try:
    from rich.console import Console as _Console
    from rich.theme import Theme as _Theme
    _con = _Console(
        highlight=False,
        theme=_Theme({
            "ok":   "bold green",
            "err":  "bold red",
            "warn": "bold yellow",
            "head": "bold",
            "muted": "dim",
            "cmd":  "bold cyan",
        }),
    )
    _HAS_RICH = True
except ImportError:
    _con = None  # type: ignore[assignment]
    _HAS_RICH = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    if _HAS_RICH:
        _con.print()
        _con.rule(f"[head]{text}[/head]")
    else:
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)


def _ok(msg: str) -> None:
    if _HAS_RICH:
        _con.print(f"[ok]✓[/ok]  {msg}")
    else:
        print(f"  ✓ {msg}")


def _err(msg: str) -> None:
    if _HAS_RICH:
        _con.print(f"[err]✗[/err]  {msg}")
    else:
        print(f"  ✗ {msg}")


def _warn(msg: str) -> None:
    if _HAS_RICH:
        _con.print(f"[warn]⚠[/warn]  {msg}")
    else:
        print(f"  ⚠ {msg}")


def _info(msg: str) -> None:
    if _HAS_RICH:
        _con.print(f"  [muted]{msg}[/muted]")
    else:
        print(f"  {msg}")


def _ask(prompt: str, default: str = "") -> str:
    """Prompt user for input, showing a default if available."""
    if default:
        display = f"{prompt} [{default}]: "
    else:
        display = f"{prompt}: "
    value = input(display).strip()
    return value if value else default


def _confirm(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw  = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Vault path
# ─────────────────────────────────────────────────────────────────────────────

def configure_vault() -> Path:
    _banner("Step 1 — Configure Obsidian vault path")
    print(
        "Enter the full path to your Obsidian vault (the root folder).\n"
        "Example: C:\\Users\\you\\Google Drive\\My Drive\\Journal\n"
    )

    while True:
        path_str = _ask("Vault path")
        if not path_str:
            print("Path cannot be empty.")
            continue

        vault_path = Path(path_str)
        if not vault_path.exists():
            _err(f"Path does not exist: {vault_path}")
            retry = _confirm("  Try a different path?", default=True)
            if not retry:
                sys.exit("Setup cancelled.")
            continue

        if not vault_path.is_dir():
            _err("Path is not a directory.")
            continue

        _ok(f"Vault found: {vault_path}")
        break

    # Save to config.json — we do this before installing deps so the config
    # module works immediately after setup.
    import json
    config_file = BASE_DIR / "config.json"
    cfg = {}
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8-sig") as f:
            cfg = json.load(f)
    cfg["vault_path"] = str(vault_path)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    _info(f"Saved to {config_file}")

    return vault_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies() -> None:
    _banner("Step 2 — Install Python dependencies")

    requirements_file = BASE_DIR / "requirements.txt"
    if not requirements_file.exists():
        _err(f"requirements.txt not found at {requirements_file}")
        sys.exit(1)

    # Install CPU-only torch first to avoid the 2+ GB GPU wheel.
    _info("Installing CPU-only torch …\n")
    torch_result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ],
        text=True,
    )
    if torch_result.returncode != 0:
        _warn("Could not install CPU-only torch. Continuing — pip will choose.")

    _info(f"Running: pip install -r {requirements_file}\n")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        text=True,
    )
    if result.returncode != 0:
        _err("pip install failed. Fix the errors above and re-run setup.py.")
        sys.exit(result.returncode)
    print()
    _ok("Dependencies installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Initial index
# ─────────────────────────────────────────────────────────────────────────────

def initial_index(vault_path: Path) -> None:
    _banner("Step 3 — Initial vault index")
    print(
        "  This will embed all your journal entries locally.\n"
        "  First run downloads the embedding model (~90 MB).\n"
        "  Subsequent runs are fast.\n"
    )

    if not _confirm("  Start indexing now?", default=True):
        _info("Skipped. Run `python indexer.py` later to index the vault.")
        return

    # Import here — deps should now be installed
    try:
        from indexer import index_vault
    except ImportError as exc:
        _err(f"Could not import indexer: {exc}")
        _info("Make sure dependencies installed correctly (Step 2).")
        sys.exit(1)

    stats = index_vault(vault_path=vault_path)
    _ok(f"Indexed {stats['files']} files into {stats['chunks']} chunks.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Register Windows Task Scheduler job
# ─────────────────────────────────────────────────────────────────────────────

def _find_pythonw() -> Path:
    """
    Return the path to pythonw.exe (windowless Python).
    Falls back to the regular interpreter if not found.
    """
    python_exe  = Path(sys.executable)
    pythonw_exe = python_exe.parent / "pythonw.exe"
    return pythonw_exe if pythonw_exe.exists() else python_exe


def _register_via_startup_folder(pythonw: Path, watcher: Path) -> bool:
    """Create a silent .vbs launcher in the user Startup folder."""
    try:
        startup = (
            Path(os.environ["APPDATA"])
            / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        )
        startup.mkdir(parents=True, exist_ok=True)
        vbs_path = startup / "JournalRAG-Watcher.vbs"
        vbs_content = (
            'Set WshShell = CreateObject("WScript.Shell")\n'
            f'WshShell.Run """{pythonw}"" ""{watcher}""", 0, False\n'
        )
        vbs_path.write_text(vbs_content, encoding="utf-8")
        _ok(f"Startup entry created: {vbs_path}")
        _info("The watcher will start silently on every login.")
        return True
    except Exception as exc:
        _err(f"Could not create startup entry: {exc}")
        _info(f"Start the watcher manually: python \"{watcher}\"")
        return False


def _start_watcher_now(pythonw: Path, watcher: Path) -> None:
    """Launch watcher.py immediately as a detached background process."""
    try:
        DETACHED        = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(
            [str(pythonw), str(watcher)],
            creationflags=DETACHED | CREATE_NO_WINDOW,
            close_fds=True,
        )
        _ok("Watcher started in the background.")
    except Exception as exc:
        _err(f"Could not start watcher: {exc}")
        _info(f"Start it manually: python \"{watcher}\"")


def register_watcher_task() -> None:
    _banner("Step 4 — Register background watcher")
    print(
        "  This registers watcher.py to start silently on every login.\n"
        "  Tries Task Scheduler first; falls back to the Startup folder.\n"
    )

    if not _confirm("  Register watcher task?", default=True):
        _info("Skipped. To start the watcher manually: python watcher.py")
        return

    pythonw   = _find_pythonw()
    watcher   = BASE_DIR / "watcher.py"
    task_name = "JournalRAG-Watcher"
    task_cmd  = f'"{pythonw}" "{watcher}"'

    schtasks_args = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", task_cmd,
        "/sc", "onlogon",
        "/rl", "limited",
        "/f",
    ]

    _info(f"Trying Task Scheduler (task: '{task_name}') …")
    result = subprocess.run(schtasks_args, capture_output=True, text=True)

    if result.returncode == 0:
        _ok(f"Task '{task_name}' registered successfully.")
        _info(f"Interpreter : {pythonw}")
        _info(f"Script      : {watcher}")
        _info(f"Trigger     : On logon")
        if _confirm("  Start the watcher right now?", default=True):
            start_result = subprocess.run(
                ["schtasks", "/run", "/tn", task_name],
                capture_output=True, text=True,
            )
            if start_result.returncode == 0:
                _ok("Watcher started in the background.")
            else:
                _err(f"Could not start task: {start_result.stderr.strip()}")
    else:
        _warn("Task Scheduler registration failed (likely needs admin rights).")
        _info("Falling back to Startup folder …")
        registered = _register_via_startup_folder(pythonw, watcher)
        if registered and _confirm("  Start the watcher right now?", default=True):
            _start_watcher_now(pythonw, watcher)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Create journal.bat
# ─────────────────────────────────────────────────────────────────────────────

def create_bat_file() -> None:
    _banner("Step 5 — Create journal.bat launcher")
    bat_path = BASE_DIR / "journal.bat"
    bat_content = (
        "@echo off\n"
        f'python "{BASE_DIR / "main.py"}" %*\n'
    )
    bat_path.write_text(bat_content, encoding="utf-8")
    _ok(f"Created: {bat_path}")
    _info(
        "To run `journal` from anywhere, add this directory to your PATH:\n"
        f"    {BASE_DIR}\n\n"
        "  Or copy journal.bat to a folder already on your PATH."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if _HAS_RICH:
        from rich.panel import Panel
        _con.print()
        _con.print(
            Panel(
                "[head]journal-rag[/head]  —  First-time setup\n"
                "[muted]This wizard will configure your local journal RAG system.\n"
                "You only need to run this once.[/muted]",
                border_style="bright_black",
                padding=(0, 1),
            )
        )
        _con.print()
    else:
        print("\n" + "=" * 60)
        print("  journal-rag — First-time setup")
        print("=" * 60)
        print(
            "\nThis wizard will configure your local journal RAG system.\n"
            "You only need to run this once.\n"
        )

    # Step 1 — vault path (no deps required)
    vault_path = configure_vault()

    # Step 2 — install deps (before importing anything that needs them)
    install_dependencies()

    # Step 3 — index
    initial_index(vault_path)

    # Step 4 — task scheduler
    register_watcher_task()

    # Step 5 — bat launcher
    create_bat_file()

    # Done
    _banner("Setup complete!")
    if _HAS_RICH:
        _con.print()
        _con.print('  [head]Usage[/head]')
        _con.print('    [cmd]journal[/cmd] [muted]"What have I been struggling with lately?"[/muted]')
        _con.print('    [cmd]journal chat[/cmd]')
        _con.print('    [cmd]journal --multi[/cmd] [muted]"What are my biggest patterns this year?"[/muted]')
        _con.print()
        _con.print('  [head]Other commands[/head]')
        _con.print('    [cmd]python indexer.py[/cmd]          [muted]— Re-index the entire vault[/muted]')
        _con.print('    [cmd]python retriever.py "q"[/cmd]    [muted]— Test retrieval without Claude[/muted]')
        _con.print('    [cmd]python main.py --dry-run "q"[/cmd] [muted]— Preview prompt without calling Claude[/muted]')
        _con.print()
        _con.print(f'  [head]Logs[/head]  [muted]{BASE_DIR / "journal-rag.log"}[/muted]')
        _con.print()
    else:
        print(
            "\n  Usage:\n"
            "    journal \"What have I been struggling with lately?\"\n"
            "    python main.py \"What were my goals at the start of the year?\"\n"
            "\n"
            "  Other commands:\n"
            "    python indexer.py          — Re-index the entire vault\n"
            "    python retriever.py \"q\"    — Test retrieval without Claude\n"
            "    python main.py --dry-run \"q\" — Preview prompt without calling Claude\n"
            "\n"
            "  Logs:\n"
            f"    {BASE_DIR / 'journal-rag.log'}\n"
        )


if __name__ == "__main__":
    main()
