"""
agents.py — Multi-agent analysis pipeline.

Runs three specialist analysts (therapist, life advisor, critic) in parallel,
then feeds all three outputs to a synthesiser that produces the final response.

Flow:
  retrieved chunks + question
      ├─── Therapist   ──┐
      ├─── Life Advisor ──┤ (parallel)
      └─── Critic      ──┘
                          └─→ Synthesiser ──→ stdout (streamed)
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prompt_builder import format_context_block
from retriever import RetrievedChunk
from ui import console

try:
    import msvcrt as _msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False

_CTRL_O = b'\x0f'

_D = "=" * 72   # divider

_AGENT_ICONS = {
    "Therapist":    "♡",   # empathy, emotional patterns
    "Life Advisor": "◎",   # goals, focus, leverage
    "Critic":       "⊘",   # negation, blind spots
}

# ─────────────────────────────────────────────────────────────────────────────
# Personas
# ─────────────────────────────────────────────────────────────────────────────

_THERAPIST = """\
You are a sharp, empathetic therapist reading a client's private journal entries.
Surface the emotional and psychological patterns underneath the surface narrative.

Focus on:
- Emotions that are present but understated, deflected, or unacknowledged
- Recurring psychological patterns (avoidance, rumination, attachment dynamics)
- What their behaviour toward people reveals about their inner needs
- Contradictions between how they describe their feelings and how they actually act
- What they seem to be carrying that they haven't directly named

Be direct. Ground every observation in the actual text — quote or paraphrase.
Write 3–5 short paragraphs. No preamble, no sign-off, no labels."""

_ADVISOR = """\
You are a clear-eyed life advisor reading a client's private journal entries.
Assess how they are actually operating in the world and where the real leverage is.

Focus on:
- The gap between stated goals and time/energy actually directed toward them
- Decision and habit patterns — especially the ones that repeat across entries
- What is working for them vs. what keeps costing them
- Where they are the bottleneck in their own life
- The 1–2 concrete shifts with the highest likely impact on their trajectory

Be specific and reference the text. No motivational filler, no generic advice.
Write 3–5 short paragraphs. No preamble, no sign-off, no labels."""

_CRITIC = """\
You are an unflinching critic reading a client's private journal entries.
Name the blind spots, distortions, and self-deceptions in how they see themselves.

Focus on:
- Where the self-narrative is flattering, selective, or lets them off the hook
- Things they consistently avoid naming or confronting directly
- Patterns they notice in others that they display themselves
- Where their framing of a situation is convenient rather than accurate
- What a genuinely neutral observer would say that they wouldn't want to hear

Ground every challenge in specific text. Be honest, not cruel.
Write 3–5 short paragraphs. No preamble, no sign-off, no labels."""

_SYNTHESISER = """\
You have three independent analyses of the same person's journal entries, plus the original \
entries themselves. Write a single response addressed directly to the person — warm, honest, \
and grounded in what they actually wrote.

Structure your response in two parts, but do NOT use headers or labels — let it flow naturally:

Part 1 — Day-to-day acknowledgment (2–3 short paragraphs):
Pick out 2–3 specific things from the entries that deserve a direct response — not as evidence \
for a pattern, but as things worth noticing in their own right. A moment they described, \
something they said in passing, a small win or a hard day. Respond to them like a trusted \
friend who read what they wrote: warm, present, specific.

Part 2 — Patterns and insight (3–4 paragraphs):
Draw on the three analyses to surface what is genuinely worth the person's attention. \
- Where all three converge, that is a high-confidence signal — anchor the response there \
- Where they create tension or disagree, name it — that tension is often the most revealing thing \
- Weave insights together; do NOT summarise each analyst in turn \
- Reference the journal text when it sharpens a point \
- End with 1–2 concrete things the person should sit with or act on — specific, not generic

Tone: direct and honest, but kind. Not a clinical report. Not a pep talk either. \
Write like someone who paid close attention and genuinely cares about the answer. \
No preamble, no sign-off, no meta-commentary about the analysts."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agent_prompt(persona: str, context_block: str, question: str) -> str:
    return (
        f"{persona}\n\n"
        f"{_D}\nJOURNAL EXCERPTS\n{_D}\n\n"
        f"{context_block}\n\n"
        f"{_D}\nQUESTION / FOCUS\n{_D}\n\n"
        f"{question}"
    )


def _synthesis_prompt(
    context_block: str,
    question: str,
    therapist: str,
    advisor: str,
    critic: str,
    instruct: str | None = None,
) -> str:
    instruct_block = (
        f"\n\n{_D}\nADDITIONAL INSTRUCTIONS\n{_D}\n\n{instruct}"
        if instruct else ""
    )
    return (
        f"{_SYNTHESISER}\n\n"
        f"{_D}\nORIGINAL JOURNAL EXCERPTS\n{_D}\n\n"
        f"{context_block}\n\n"
        f"{_D}\nQUESTION ASKED\n{_D}\n\n"
        f"{question}{instruct_block}\n\n"
        f"{_D}\nTHERAPIST\n{_D}\n\n"
        f"{therapist}\n\n"
        f"{_D}\nLIFE ADVISOR\n{_D}\n\n"
        f"{advisor}\n\n"
        f"{_D}\nCRITIC\n{_D}\n\n"
        f"{critic}"
    )


def _call_claude_capture(prompt: str, model: str) -> str:
    """Run claude --print, capture and return the text output."""
    result = subprocess.run(
        ["claude", "--print", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        err = result.stderr.strip() or "non-zero exit"
        return f"[error: {err}]"
    return result.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive agent-output viewer  (Ctrl+O cycles expanded panels)
# ─────────────────────────────────────────────────────────────────────────────

def _show_agent_outputs(responses: dict[str, str], agent_names: list[str]) -> None:
    """
    Display individual agent analyses interactively.

    Ctrl+O  — expand next agent (cycles: collapsed → Therapist → Advisor → Critic → collapsed)
    Enter / Space / Esc  — exit and continue to synthesis
    """
    def _panel(name: str, expanded: bool) -> Panel:
        glyph = _AGENT_ICONS.get(name, "◆")
        text  = responses.get(name, "")
        if expanded:
            return Panel(
                text,
                title=f"[bold cyan]  {glyph}  {name}  [/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        lines   = [ln for ln in text.split('\n') if ln.strip()]
        preview = (lines[0][:70] + " …") if lines else "…"
        return Panel(
            f"[dim]{preview}[/dim]",
            title=f"[dim]  {glyph}  {name}  [/dim]",
            border_style="bright_black",
            padding=(0, 1),
        )

    def _render(fi: int) -> Group:
        panels = [_panel(name, fi == i) for i, name in enumerate(agent_names)]
        if fi < 0:
            hint = "  [dim]Ctrl+O[/dim]  [muted]expand  ·  [dim]Enter[/dim]  [muted]continue to synthesis[/muted]"
        else:
            glyph = _AGENT_ICONS.get(agent_names[fi], "◆")
            hint  = (
                f"  [dim]Ctrl+O[/dim]  [muted]next  ·  [dim]Enter[/dim]  [muted]continue  "
                f"·  {glyph} [cyan]{agent_names[fi]}[/cyan]"
            )
        return Group(*panels, Text.from_markup(hint))

    console.print()

    if not _HAS_MSVCRT:
        # Non-interactive fallback: print all panels sequentially
        for name in agent_names:
            console.print(_render(-1))
        console.print()
        return

    focused = -1
    with Live(_render(focused), console=console, refresh_per_second=10) as live:
        while True:
            if _msvcrt.kbhit():
                key = _msvcrt.getch()
                if key in (b'\x00', b'\xe0'):   # extended key prefix — swallow next byte
                    _msvcrt.getch()
                    continue
                if key == _CTRL_O:
                    focused = focused + 1 if focused + 1 < len(agent_names) else -1
                    live.update(_render(focused))
                elif key == b'\x03':            # Ctrl+C
                    raise KeyboardInterrupt
                elif key in (b'\r', b' ', b'q', b'\x1b'):
                    break
            time.sleep(0.02)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_multi_agent(
    question: str,
    chunks: list[RetrievedChunk],
    model: str,
    synthesis_model: str,
    instruct: str | None = None,
    capture_output: bool = False,
) -> int | str:
    """
    Run the three-analyst pipeline.

    Args:
        model:           Model used for the three parallel agents.
        synthesis_model: Model used for the final synthesiser.
        capture_output:  When True (chat mode), captures and returns synthesis
                         text instead of streaming it; returns "" on failure.
                         When False (one-shot), streams synthesis and returns
                         the subprocess exit code.
    """
    context_block = format_context_block(chunks)

    agents = [
        ("Therapist",    _THERAPIST),
        ("Life Advisor", _ADVISOR),
        ("Critic",       _CRITIC),
    ]

    console.print(f"\n  [head]Running {len(agents)} analysts in parallel[/head] [muted][{model}][/muted] …\n")

    responses: dict[str, str] = {}
    statuses:  dict[str, str] = {name: "running" for name, _ in agents}

    def _make_content() -> Group:
        t    = Table(show_header=False, box=None, padding=(0, 1), show_edge=False)
        done = 0
        for agent_name, state in statuses.items():
            glyph = _AGENT_ICONS.get(agent_name, "◆")
            if state == "running":
                icon  = f"[dim]{glyph}[/dim]"
                label = f"[muted]{agent_name}[/muted]"
                note  = "[info]running …[/info]"
            elif state == "done":
                icon  = f"[bold cyan]{glyph}[/bold cyan]"
                label = agent_name
                note  = "[muted]done[/muted]"
                done += 1
            else:
                icon  = "[err]✗[/err]"
                label = f"[err]{agent_name}[/err]"
                note  = f"[err]{state}[/err]"
                done += 1
            t.add_row(f"  {icon}", label, note)
        total  = len(agents)
        bar_w  = 24
        filled = int(done / total * bar_w) if total else 0
        color  = "green" if done == total else "cyan"
        bar    = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * (bar_w - filled)}[/dim]"
        prog   = Text.from_markup(f"\n  {bar}  [muted]{done} / {total}[/muted]")
        return Group(t, prog)

    with Live(_make_content(), console=console, refresh_per_second=10) as live:
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {
                pool.submit(
                    _call_claude_capture,
                    _agent_prompt(persona, context_block, question),
                    model,
                ): name
                for name, persona in agents
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    responses[name] = future.result()
                    statuses[name]  = "done"
                except Exception as exc:
                    responses[name] = f"[failed: {exc}]"
                    statuses[name]  = str(exc)
                live.update(_make_content())

    # ── Interactive agent-output viewer ────────────────────────────────────
    _show_agent_outputs(responses, [name for name, _ in agents])

    synth_prompt = _synthesis_prompt(
        context_block = context_block,
        question      = question,
        therapist     = responses.get("Therapist",    "[no response]"),
        advisor       = responses.get("Life Advisor", "[no response]"),
        critic        = responses.get("Critic",       "[no response]"),
        instruct      = instruct,
    )

    console.print(f"  [head]✦  Synthesising[/head] [muted][{synthesis_model}][/muted] …\n")
    console.rule(style="bright_black")
    console.print()

    if capture_output:
        result = subprocess.run(
            ["claude", "--print", "--model", synthesis_model],
            input=synth_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        text = result.stdout.strip()
        console.print(text)
        console.print()
        return text if result.returncode == 0 else ""

    result = subprocess.run(
        ["claude", "--print", "--model", synthesis_model],
        input=synth_prompt,
        text=True,
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    )
    return result.returncode
