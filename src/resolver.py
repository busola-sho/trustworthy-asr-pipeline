"""
src/resolver.py

Sentence-level resolver for ASR disagreements after ROVER combination.

Two mechanisms:
1. Lexicon scan — find Scottish words in model outputs missing from ROVER
2. LLM pass — only for risky disagreements (negation/number/named_entity)
   with full sentence context from all models + specific flagged positions

Usage:
    from src.resolver import resolve_sentence

    resolved, log = resolve_sentence(
        rover_transcript="she was near the pub",
        model_sentences={
            "qwen":    "she was near the pub",
            "whisper": "she wisnae near the pub",
            ...
        },
        disagreements=[...],
        client=client,
    )
"""

import time
from typing import List, Dict, Optional, Tuple
from ollama import Client

OLLAMA_HOST    = "http://localhost:11434"
RESOLVER_MODEL = "qwen2.5:7b"

# ── Scottish lexicon ───────────────────────────────────────────────────────────

SCOTTISH_NEGATIONS = {
    "wisnae", "isnae", "isna", "cannae", "canna", "dinnae", "dinna",
    "didnae", "didna", "hasnae", "hasna", "havnae", "wouldnae", "couldnae",
    "shouldnae", "willnae", "wasnae", "nae", "naw",
}

SCOTTISH_DIALECT = {
    "wee", "braw", "aye", "och", "dreich", "ken", "tae", "fae", "oot",
    "doon", "bairn", "loch", "burn", "awa", "aboot", "hoose", "noo",
    "fer", "wi", "wae", "guid", "mair", "dae", "hae", "tak", "gi", "gie",
    "bairns", "lassie", "laddie", "bonnie", "haar", "muckle", "weans",
    "outwith", "thae", "whit", "aff", "twa", "haud", "bide", "lang",
    "auld", "cauld", "fou", "gey", "heid", "lugs", "polis", "stey",
}

SCOTTISH_LEXICON = SCOTTISH_NEGATIONS | SCOTTISH_DIALECT


def _norm(word: str) -> str:
    return word.lower().strip(".,!?;:'\"()[]") if word else ""


def _is_scottish(word: str) -> bool:
    return _norm(word) in SCOTTISH_LEXICON


# ── Lexicon scan ───────────────────────────────────────────────────────────────

def lexicon_notes(
    rover_transcript: str,
    model_sentences: Dict[str, str],
) -> str:
    """
    Find Scottish words present in model outputs but missing from ROVER output.
    Includes surrounding context so the LLM knows where to restore them.
    Returns a formatted string for the LLM prompt, or empty string if none.
    """
    rover_words = set(_norm(w) for w in rover_transcript.split())
    notes = []
    seen  = set()

    for model, text in model_sentences.items():
        words = text.split()
        for i, word in enumerate(words):
            normed = _norm(word)
            if _is_scottish(word) and normed not in rover_words and normed not in seen:
                # get surrounding context (±2 words)
                left  = " ".join(words[max(0, i-2):i])
                right = " ".join(words[i+1:i+3])
                ctx   = f"...{left} [{word}] {right}...".strip()
                notes.append(f"  '{word}' (from {model}) — appears as: \"{ctx}\"")
                seen.add(normed)

    return "\n".join(notes) if notes else ""


# ── LLM resolver ──────────────────────────────────────────────────────────────

RESOLVER_PROMPT = """You are correcting an ASR transcript of a Scottish English police interview.

ROVER combination produced this draft:
  {rover_transcript}

Original model outputs for this sentence:
{model_outputs}

Specific disagreements to resolve:
{flagged_issues}
{scottish_note}
Rules:
- You may ONLY use words that appear VERBATIM in one of the model outputs listed above
- Do NOT add, invent, or generate ANY words from your own knowledge — not even Scottish words
- ONLY use words like "wisnae", "cannae", "dinnae" if they appear VERBATIM in a model output above
- Pay close attention to the flagged disagreements — focus corrections there
- CRITICAL: Negation errors reverse meaning entirely in policing. Only correct a negation if the alternative word appears in a model output above.

Return ONLY the corrected sentence. No explanation, no preamble."""


def _format_model_outputs(model_sentences: Dict[str, str]) -> str:
    lines = []
    for model, text in model_sentences.items():
        if text and text.strip():
            lines.append(f"  {model}: {text.strip()}")
    return "\n".join(lines)


def _format_flagged_issues(disagreements: List[dict]) -> str:
    """Format risky disagreements as specific instructions for the LLM."""
    risky = [d for d in disagreements
             if d["category"] in ("negation", "number", "named_entity")]

    if not risky:
        return "  (none flagged — focus on Scottish word restoration)"

    lines = []
    for d in risky:
        words = {m: w for m, w in d["hypotheses"].items() if w and w.strip()}
        cat   = d["category"].upper()
        lines.append(f"  [{cat}] Models disagree: {words}")

    return "\n".join(lines)


def llm_resolve_sentence(
    rover_transcript: str,
    model_sentences: Dict[str, str],
    disagreements: List[dict],
    client: Client,
    model: str = RESOLVER_MODEL,
) -> Optional[str]:
    """Call LLM to resolve a sentence. Returns corrected text or None."""

    model_outputs_str = _format_model_outputs(model_sentences)
    flagged_str       = _format_flagged_issues(disagreements)
    scottish_str      = lexicon_notes(rover_transcript, model_sentences)

    scottish_note = ""
    if scottish_str:
        scottish_note = (
            "\nScottish words in model outputs MISSING from ROVER "
            "(strongly consider restoring):\n" + scottish_str + "\n"
        )

    prompt = RESOLVER_PROMPT.format(
        rover_transcript=rover_transcript,
        model_outputs=model_outputs_str,
        flagged_issues=flagged_str,
        scottish_note=scottish_note,
    )

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 2048},
        )
        resolved = response.message.content.strip()

        if not resolved:
            return None

        # sanity check: reject if wildly longer than input
        if len(resolved.split()) > len(rover_transcript.split()) * 2:
            print(f"    REJECTED — output too long ({len(resolved.split())} words)")
            return None

        return resolved

    except Exception as e:
        print(f"  ERROR (llm_resolver): {e}")
        return None


# ── Main entry point ───────────────────────────────────────────────────────────

def resolve_sentence(
    rover_transcript: str,
    model_sentences: Dict[str, str],
    disagreements: List[dict],
    client: Client,
    model: str = RESOLVER_MODEL,
    sleep: float = 0.05,
) -> Tuple[str, dict]:
    """
    Resolve one sentence.

    Triggers LLM only when:
    - There are risky disagreements (negation/number/named_entity), OR
    - Scottish words appear in model outputs but are missing from ROVER output

    Args:
        rover_transcript: sentence produced by ROVER voting
        model_sentences:  {model: sentence_text} for this segment
        disagreements:    disagreement dicts from ROVERResult
        client:           Ollama client
        model:            Ollama model name

    Returns:
        (resolved_transcript, log_dict)
    """
    risky    = [d for d in disagreements
                if d["category"] in ("negation", "number", "named_entity")]

    # only call LLM if there are genuine risky disagreements
    # Scottish word notes are hints only — not a trigger
    should_call_llm = len(risky) > 0

    if not should_call_llm:
        return rover_transcript, {
            "called_llm": False,
            "rover":      rover_transcript,
            "resolved":   rover_transcript,
            "changed":    False,
            "n_risky":    0,
        }

    scottish = lexicon_notes(rover_transcript, model_sentences)

    resolved = llm_resolve_sentence(
        rover_transcript, model_sentences, disagreements, client, model
    )
    time.sleep(sleep)

    if resolved is None:
        return rover_transcript, {
            "called_llm": True,
            "rover":      rover_transcript,
            "resolved":   rover_transcript,
            "changed":    False,
            "error":      True,
        }

    changed = resolved.strip() != rover_transcript.strip()
    if changed:
        print(f"  [resolver] ROVER: {rover_transcript[:80]}")
        print(f"  [resolver]  OUT:  {resolved[:80]}")

    return resolved, {
        "called_llm":     True,
        "rover":          rover_transcript,
        "resolved":       resolved,
        "changed":        changed,
        "scottish_notes": scottish,
        "n_risky":        len(risky),
    }


def check_resolver_available(client: Client, model: str = RESOLVER_MODEL) -> bool:
    try:
        available = [m.model for m in client.list().models]
        return any(model in m for m in available)
    except Exception:
        return False