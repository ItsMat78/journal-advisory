"""
prompt_builder.py — Assembles the full prompt that gets piped into Claude Code.

The prompt includes:
  - A system persona (reflective journal coach)
  - Retrieved journal excerpts, formatted with metadata
  - The user's question
"""

from retriever import RetrievedChunk

# ─────────────────────────────────────────────────────────────────────────────
# System persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PERSONA = """\
You are a thoughtful companion reading the user's personal journal entries. \
Your role is to respond to what they've actually written — both the specific \
day-to-day things they mention and the broader patterns underneath.

How you operate:
- Acknowledge the specific things they wrote about first — a moment, a feeling, \
  something they said in passing. Respond to it directly before stepping back to analyse.
- Be honest, not sycophantic. The user is not looking for empty validation; they want \
  someone who actually read what they wrote and has something real to say.
- Ground every observation in the actual text. Quote or paraphrase specific entries \
  when it strengthens your point.
- Name patterns plainly when they are genuinely present — including uncomfortable ones.
- If the provided context is insufficient to answer confidently, say so rather than \
  speculating.
- Tone: warm and direct. Like a trusted friend who paid close attention — not a \
  clinician filing a report, not a life coach delivering a framework.
- Treat all journal content with discretion — this is private material.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _format_file_type_label(file_type: str) -> str:
    labels = {
        "daily":  "Daily Journal",
        "people": "People Note",
        "note":   "Note",
    }
    return labels.get(file_type, "Entry")


def _format_chunk(chunk: RetrievedChunk, index: int) -> str:
    """Format a single retrieved chunk into a readable block."""
    label = _format_file_type_label(chunk.file_type)
    header = f"[{index}] {label} — {chunk.date} (source: {chunk.source})"
    separator = "─" * len(header)
    return f"{header}\n{separator}\n{chunk.text}"


def format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format all retrieved chunks into a single context block."""
    if not chunks:
        return "(No relevant journal entries found for this query.)"

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(_format_chunk(chunk, i))

    return "\n\n".join(formatted)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    instruct: str | None = None,
) -> str:
    """
    Construct the complete prompt to be piped into Claude Code.

    Structure:
      SYSTEM PERSONA
      ──────────────
      RETRIEVED JOURNAL EXCERPTS
      ──────────────
      USER QUESTION
      ──────────────  (optional)
      ADDITIONAL INSTRUCTIONS
    """
    context_block = format_context_block(chunks)

    divider = "=" * 72

    instruct_block = (
        f"\n{divider}\nADDITIONAL INSTRUCTIONS\n{divider}\n\n{instruct}\n"
        if instruct else ""
    )

    prompt = f"""\
{_SYSTEM_PERSONA}

{divider}
RELEVANT JOURNAL EXCERPTS
({len(chunks)} excerpt(s) retrieved — ordered by relevance)
{divider}

{context_block}

{divider}
USER QUESTION
{divider}

{question}
{instruct_block}
{divider}

Please respond based on the journal excerpts above. \
If multiple entries are relevant, synthesise across them. \
Be specific and honest.
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from retriever import retrieve

    if len(sys.argv) < 2:
        print("Usage: python prompt_builder.py \"your question\"")
        sys.exit(1)

    q      = " ".join(sys.argv[1:])
    chunks = retrieve(q)
    prompt = build_prompt(q, chunks)
    print(prompt)
