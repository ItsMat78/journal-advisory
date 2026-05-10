<div align="center">

```
     ██╗ ██████╗ ██╗   ██╗██████╗ ███╗   ██╗ █████╗ ██╗     
     ██║██╔═══██╗██║   ██║██╔══██╗████╗  ██║██╔══██╗██║     
     ██║██║   ██║██║   ██║██████╔╝██╔██╗ ██║███████║██║     
██   ██║██║   ██║██║   ██║██╔══██╗██║╚██╗██║██╔══██║██║     
╚█████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║██║  ██║███████╗
 ╚════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
          ___       __      _                      
         /   | ____/ /   __(_)________  _______  __
        / /| |/ __  / | / / / ___/ __ \/ ___/ / / /
       / ___ / /_/ /| |/ / (__  ) /_/ / /  / /_/ / 
      /_/  |_\__,_/ |___/_/____/\____/_/   \__, /  
                                          /____/    
```

**Your personal journal, analyzed.**

*Indexes your Obsidian vault locally. Retrieves what's relevant.  
Runs it through Claude. Tells you the truth.*

---

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)
![Claude](https://img.shields.io/badge/Powered%20by-Claude%20Code%20CLI-D97706?style=flat-square)
![Local](https://img.shields.io/badge/Embeddings-Local%20only-22C55E?style=flat-square)

</div>

<br>

Ask a question. Get an honest answer drawn from your own words — not a pep talk, not speculation. Every response is grounded in what you actually wrote.

Embeddings are computed locally on your machine. Only the final prompt (your question + retrieved excerpts) is ever sent to Claude.

---

## How it works

```
Your question
     │
     ▼
sentence-transformers  ──  local embedding model (all-MiniLM-L6-v2, ~90 MB, cached)
     │
     ▼
ChromaDB  ──  semantic search + recency weighting across your vault chunks
     │
     ▼
prompt_builder.py  ──  wraps context + question (+ optional instruction) into a prompt
     │
     ▼
claude --print  ──  Claude Code CLI, running locally
     │
     ▼
Response printed to terminal
```

---

## Installation

### Requirements

| | |
|---|---|
| Python 3.11+ | `python --version` |
| Claude Code CLI | `claude --version` · install at [claude.ai/code](https://claude.ai/code) |
| Obsidian vault on disk | Synced via Google Drive, OneDrive, or local — doesn't matter |

### First-time setup

```cmd
python setup.py
```

The wizard handles everything in five steps:

1. **Vault path** — points the system at your Obsidian root folder
2. **Dependencies** — runs `pip install -r requirements.txt`
3. **Indexing** — embeds your entire vault into a local ChromaDB database  
   *(first run downloads the embedding model ~90 MB and takes a few minutes)*
4. **Watcher** — registers a Windows Task Scheduler job that auto-indexes file changes on login
5. **Launcher** — creates `journal.bat` so you can type `journal` from any terminal

### Add to PATH (run `journal` from anywhere)

**Win + S → "Environment Variables" → User Variables → Path → Edit → New**

Paste the full path to this folder, e.g. `C:\Users\you\journal-rag`. Open a fresh terminal for it to take effect.

---

## Usage

### One-shot mode

Ask a question, get a response, done.

```cmd
journal "What have I been stressed about lately?"
journal "How has my relationship with Soumya changed over time?"
journal "What were my goals at the start of the year?"
journal "When did I last feel genuinely excited about something?"
```

Pull the most recent entries without a question:

```cmd
journal --latest          # most recent entry
journal --latest 3        # last 3 entries
```

---

### Chat mode

Multi-turn conversation. Fresh context is retrieved every turn. Full history is carried forward. Sessions are saved to `./sessions/` as Markdown.

```cmd
journal chat                           # start a new session
journal chat "how has my sleep been?"  # start with a first message
journal chat --resume                  # pick from recent sessions interactively
journal chat --resume latest           # auto-resume the most recent
journal chat --multi                   # every turn runs all three analysts + synthesis
```

**In-session commands:**

| Command | What it does |
|---------|-------------|
| `/latest` | Pull the most recent journal entry into context |
| `/latest 3` | Pull the 3 most recent entries |
| `/instruct don't mention work` | Set a response directive for remaining turns |
| `/instruct` | Clear the current directive |
| `exit` · `quit` · `Ctrl+C` | End session and save |

---

### Multi-agent mode

Three analysts run in parallel against the same retrieved context, then a synthesiser weaves their findings into a single unified response.

| Agent | What it looks for |
|-------|------------------|
| **Therapist** | Emotional patterns, psychological dynamics, what's unspoken or deflected |
| **Life Advisor** | Goals vs. actual time/energy, habit patterns, highest-leverage shifts |
| **Critic** | Blind spots, self-deceptions, the things you don't want to hear |

```cmd
journal --multi "What are my biggest patterns this year?"
journal --multi --latest
journal --multi --deep "Give me a full psychological profile"
```

After the three analyses complete, they appear as **collapsible panels**:

- **`Ctrl+O`** — cycle through agents, expanding one at a time  
- **`Enter` / `Space`** — continue to the synthesised response

This works identically in chat mode (`journal chat --multi`), where the synthesis is also saved to session history.

---

### All flags

| Flag | Modes | Description |
|------|-------|-------------|
| `--latest [N]` | one-shot, multi | Use the N most recent entries instead of searching |
| `--since YYYY-MM-DD` | all | Only search entries on or after this date |
| `--top-k N` | all | Chunks to retrieve per question (default: 15) |
| `--deep` | all | Use `claude-opus-4-6` for the final response (Max subscription required) |
| `--multi` | one-shot, chat | Run three analysts + synthesiser instead of a single response |
| `--instruct "text"` | all | Extra directive for Claude — does **not** affect which chunks are retrieved |
| `--resume [latest\|stem]` | chat | Resume a previous session |
| `--dry-run` | one-shot | Print the assembled prompt without calling Claude |

---

### The `--instruct` flag

Adds a directive Claude must follow in its response, without touching retrieval. The question still drives what's retrieved — `--instruct` only shapes the reply.

```cmd
journal "what's been weighing on me?" --instruct "don't mention work stress"
journal "how am I doing with relationships?" --instruct "focus on patterns across time, not individual events"
journal chat --instruct "be extremely direct, no softening language"
journal --multi "where am I stuck?" --instruct "keep the synthesis under 300 words"
```

In chat mode, update or clear it mid-session with `/instruct`.

---

### Filter by date

```cmd
journal --since 2026-01-01 "What were my January goals?"
journal chat --since 2026-04-01
journal --multi --since 2026-05-01 "exam week patterns"
```

---

## File structure

```
journal-rag/
├── main.py            ← CLI entrypoint (one-shot + multi-agent)
├── chat.py            ← interactive multi-turn chat mode
├── agents.py          ← multi-agent pipeline (therapist · advisor · critic · synthesiser)
├── prompt_builder.py  ← assembles the prompt sent to Claude
├── retriever.py       ← semantic + recency hybrid search
├── indexer.py         ← vault scanner, chunker, embedder
├── watcher.py         ← background auto-indexer (runs on login)
├── config.py          ← all settings (models, chunk size, weights)
├── ui.py              ← terminal formatting via Rich
├── setup.py           ← first-run setup wizard
├── requirements.txt   ← Python dependencies
├── journal.bat        ← Windows launcher
│
├── db/                ← ChromaDB vector store (local, never leaves machine)
├── sessions/          ← saved chat sessions (.md files)
└── tmp/               ← temp prompt files (auto-deleted after each query)
```

---

## Configuration

Edit `config.py` to tune the system:

| Setting | Default | What it controls |
|---------|---------|-----------------|
| `TOP_K` | `15` | Chunks retrieved per query |
| `CHUNK_SIZE` | `300` | Max words per chunk |
| `CHUNK_OVERLAP` | `30` | Overlap words between consecutive chunks |
| `RECENCY_WEIGHT` | `0.3` | Recency vs. semantic balance (0 = pure semantic, 1 = pure recency) |
| `RECENCY_HALF_LIFE_DAYS` | `30` | Days until a chunk's recency score halves |
| `MODEL_DEFAULT` | `claude-sonnet-4-6` | Default model (Pro subscription) |
| `MODEL_DEEP` | `claude-opus-4-6` | Model used with `--deep` (Max subscription) |
| `SKIP_DIRS` | `{"Images", …}` | Directories never indexed |

After changing chunk settings, re-index the vault:

```cmd
python indexer.py --force
```

---

## Indexing

```cmd
python indexer.py          # re-index changed files only (fast, incremental)
python indexer.py --force  # re-embed everything from scratch
python retriever.py "q"    # test retrieval without calling Claude
```

The watcher handles incremental indexing automatically in the background. You only need to run `indexer.py` manually after a full re-embed or vault restructure.

---

## Background watcher

Monitors your vault for file changes and re-indexes modified files silently via Windows Task Scheduler.

```cmd
schtasks /query  /tn "JournalRAG-Watcher"       # check status
schtasks /run    /tn "JournalRAG-Watcher"       # start manually
schtasks /end    /tn "JournalRAG-Watcher"       # stop
schtasks /delete /tn "JournalRAG-Watcher" /f    # remove task
```

View the log:

```cmd
type journal-rag.log
```

---

## Troubleshooting

**`journal` is not recognised as a command**  
The folder isn't on your PATH yet. Either complete Setup → Add to PATH, or run `python main.py "..."` directly from inside the project folder.

**"No relevant journal entries found"**  
The vault isn't indexed. Run `python indexer.py`. If it was indexed but retrieval still fails, the question might not semantically match anything — try rephrasing or use `--latest`.

**Very slow first query**  
The embedding model downloads once (~90 MB) on first use. All subsequent queries are fast.

**ChromaDB or import errors after install**  
Close and reopen your terminal after `pip install`, then retry.

**Watcher task fails to register**  
Run the terminal as Administrator when executing `python setup.py`.

**Prompt is long and responses are slow**  
Reduce `TOP_K` in `config.py` (try 8–10). More chunks = better recall but longer prompts and slower responses.

**Multi-agent mode is slow**  
Expected — three agents run sequentially per Claude Code's concurrency limits, then synthesis runs. Use `--deep` only when you want maximum depth and have time for it.

---

## Privacy

- Embeddings are computed **locally** by `sentence-transformers` — nothing leaves your machine during indexing
- The ChromaDB database (`./db/`) stores your journal text in plain form — keep the project folder private
- Only the assembled prompt (your question + retrieved excerpts) is sent to Claude via the `claude` CLI, subject to [Anthropic's privacy policy](https://www.anthropic.com/privacy)
