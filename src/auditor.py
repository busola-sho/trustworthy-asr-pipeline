"""
src/auditor.py

Sentence-level semantic auditor — the final stage after ROVER combination.

Unlike the resolver (which patches isolated flagged word positions),
the auditor reads the FULL sentence with ROVER's draft, all 4 original
model transcripts, and word-level vote confidence. It acts as a final
semantic check for meaning-altering errors that majority voting might
have introduced — particularly negations, named entities, and numbers
where ROVER's vote could be wrong even with majority agreement.

The auditor is conservative by design: it should only change words when
there is clear evidence from the model transcripts that ROVER's vote
introduced a meaning-altering error. It must not paraphrase or rewrite.

Usage:
    from src.auditor import audit_sentence

    audited, log = audit_sentence(
        rover_transcript="I think I'd quite like to be a British sign language interpreter",
        model_sentences={...},
        word_confidences=[("I", 1.0), ("think", 0.95), ...],
        client=client,
    )
"""

import time
from typing import List, Dict, Optional, Tuple
from ollama import Client

OLLAMA_HOST   = "http://localhost:11434"
AUDITOR_MODEL = "qwen2.5:7b"

LOW_CONF_THRESHOLD = 0.7


AUDIT_PROMPT = """You are auditing an ASR transcript for a Scottish English police interview — a high-stakes setting where meaning-altering errors have serious consequences.

ROVER (a principled word-voting combination of 4 ASR models) produced this transcript:
  {rover_transcript}

The 4 original model transcripts it was built from:
{model_outputs}

Words ROVER was LESS confident about (vote was not unanimous):
{low_confidence_words}

YOUR TASK: ROVER has already done the bulk of the combination work through voting. Read the transcript and check ONLY for meaning-altering errors that the vote might have gotten wrong — especially:
1. NEGATIONS — could a "was" actually be a "wasn't" or Scottish "wisnae"? Check if the original model transcripts support this.
2. NAMED ENTITIES — names, places, organisations that look garbled or wrong
3. NUMBERS — times, dates, amounts that don't make sense in context

RULES:
- Do NOT rewrite, paraphrase, or restructure the sentence
- Do NOT fix minor grammar, filler words, or stylistic issues — ROVER's draft is fine as is for those
- Only change a word if you find STRONG evidence in the original model transcripts that supports the change AND the current word would alter meaning if wrong
- You may ONLY use words that appear in the original model transcripts above — never invent words
- If you find no meaning-altering errors, return the ROVER transcript completely unchanged

Return ONLY the final transcript (corrected or unchanged). No explanation, no preamble."""


def _format_model_outputs(model_sentences: Dict[str, str]) -> str:
    lines = []
    for model, text in model_sentences.items():
        if text and text.strip():
            lines.append(f"  {model}: {text.strip()}")
    return "\n".join(lines)


def _format_low_confidence(
    word_confidences: List[Tuple[str, float]],
    threshold: float = LOW_CONF_THRESHOLD,
) -> str:
    low_conf = [
        (i, w, c) for i, (w, c) in enumerate(word_confidences)
        if c < threshold
    ]
    if not low_conf:
        return "  (none — ROVER vote was unanimous throughout)"

    lines = []
    for i, w, c in low_conf:
        # context window
        start = max(0, i - 2)
        end   = min(len(word_confidences), i + 3)
        ctx_words = [word_confidences[j][0] for j in range(start, end)]
        ctx = " ".join(ctx_words)
        lines.append(f"  '{w}' (vote confidence: {c:.2f}) — context: \"{ctx}\"")
    return "\n".join(lines)


def audit_sentence(
    rover_transcript: str,
    model_sentences: Dict[str, str],
    word_confidences: List[Tuple[str, float]],
    client: Client,
    model: str = AUDITOR_MODEL,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
    sleep: float = 0.05,
) -> Tuple[str, dict]:
    """
    Audit a ROVER-combined sentence for residual meaning-altering errors.

    Only calls the LLM if there are words below the confidence threshold
    (i.e. ROVER's vote was not unanimous somewhere in this sentence).
    If ROVER was fully unanimous, skip the LLM call entirely — nothing
    to audit.

    Args:
        rover_transcript: ROVER's combined sentence
        model_sentences:  {model: sentence_text} — original transcripts
        word_confidences: [(word, confidence), ...] from ROVER voting
        client:            Ollama client
        model:             Ollama model name
        low_conf_threshold: confidence below which a word triggers audit

    Returns:
        (audited_transcript, log_dict)
    """
    has_low_conf = any(c < low_conf_threshold for _, c in word_confidences)

    if not has_low_conf:
        return rover_transcript, {
            "called_llm": False,
            "rover":      rover_transcript,
            "audited":    rover_transcript,
            "changed":    False,
        }

    model_outputs_str = _format_model_outputs(model_sentences)
    low_conf_str       = _format_low_confidence(word_confidences, low_conf_threshold)

    prompt = AUDIT_PROMPT.format(
        rover_transcript=rover_transcript,
        model_outputs=model_outputs_str,
        low_confidence_words=low_conf_str,
    )

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 2048},
        )
        audited = response.message.content.strip()

        if not audited:
            return rover_transcript, {
                "called_llm": True, "rover": rover_transcript,
                "audited": rover_transcript, "changed": False, "error": True,
            }

        # sanity check — reject wildly different length output
        if len(audited.split()) > len(rover_transcript.split()) * 1.5:
            print(f"    REJECTED — output too long")
            return rover_transcript, {
                "called_llm": True, "rover": rover_transcript,
                "audited": rover_transcript, "changed": False, "error": True,
            }

        changed = audited.strip() != rover_transcript.strip()
        if changed:
            print(f"  [auditor] ROVER: {rover_transcript[:80]}")
            print(f"  [auditor]   OUT: {audited[:80]}")

        time.sleep(sleep)

        return audited, {
            "called_llm": True,
            "rover":      rover_transcript,
            "audited":    audited,
            "changed":    changed,
        }

    except Exception as e:
        print(f"  ERROR (auditor): {e}")
        return rover_transcript, {
            "called_llm": True, "rover": rover_transcript,
            "audited": rover_transcript, "changed": False, "error": True,
        }


def check_auditor_available(client: Client, model: str = AUDITOR_MODEL) -> bool:
    try:
        available = [m.model for m in client.list().models]
        return any(model in m for m in available)
    except Exception:
        return False