"""
src/dialect_pass.py

Phonetic-similarity-gated Scottish dialect correction pass.

Catches cases where ALL models confidently agreed on a standard English
word that is actually a mis-heard Scottish dialect word — e.g. "clays"
when the speaker said "claes" (clothes). Since all models agree, ROVER's
confidence signal gives no hint that anything is wrong, so this pass
runs independently using phonetic similarity as its trigger instead.

Pipeline:
1. Phonetic scan — for every word in the transcript, compute Double
   Metaphone code and compare against a curated Scottish lexicon.
   Flag candidates within a similarity threshold.
2. Semantic judge — for each flagged candidate, ask an LLM (constrained,
   conservative) whether the Scottish word makes MORE sense in context.
   Default to "no change" on any uncertainty.

This pass runs AFTER ROVER + auditor, on the most refined transcript
available, so it only needs to catch residual dialect-blind errors.

Usage:
    from src.dialect_pass import run_dialect_pass

    corrected, log = run_dialect_pass(
        transcript="I especially like to spend money on clays",
        client=client,
    )
"""

import time
from typing import List, Dict, Optional, Tuple
import jellyfish
from ollama import Client

OLLAMA_HOST  = "http://localhost:11434"
JUDGE_MODEL  = "qwen2.5:7b"

# similarity threshold for phonetic matching (0-1, Jaro-Winkler on metaphone codes)
PHONETIC_THRESHOLD = 0.94
MIN_WORD_LENGTH     = 5  # skip short words — phonetic codes collapse and false-match easily

# High-frequency English function/content words that should never be flagged as
# dialect candidates, regardless of phonetic score — these are common enough that
# any "match" is virtually always coincidental code collapse, not genuine confusion.
COMMON_WORD_STOPLIST = {
    "like", "loads", "but", "can", "buy", "please", "wanted", "second",
    "could", "would", "should", "about", "around", "again", "after",
    "before", "people", "things", "think", "going", "doing", "really",
    "always", "never", "every", "money", "place", "where", "there",
    "their", "these", "those", "which", "while", "still", "right",
    "great", "small", "large", "happy", "quite", "maybe", "actually",
    # standard English negations — must never be flagged as Scottish
    # negation candidates (didnae/wisnae etc.); these are overwhelmingly
    # correct as standard English and the judge over-approves them
    "didn't", "didnt", "doesn't", "doesnt", "isn't", "isnt", "wasn't",
    "wasnt", "hasn't", "hasnt", "couldn't", "couldnt", "wouldn't",
    "wouldnt", "shouldn't", "shouldnt", "house", "looks", "born",
}

# ── Curated Scottish lexicon with meanings ─────────────────────────────────────
# Each entry: scottish_word -> standard_english_meaning

SCOTTISH_LEXICON = {
    "claes":    "clothes",
    "bairn":    "child",
    "bairns":   "children",
    "wean":     "child",
    "weans":    "children",
    "ken":      "know",
    "tae":      "to",
    "fae":      "from",
    "oot":      "out",
    "doon":     "down",
    "hoose":    "house",
    "moose":    "mouse",
    "loch":     "lake",
    "burn":     "stream",
    "wee":      "small",
    "braw":     "good/fine",
    "guid":     "good",
    "bonnie":   "beautiful",
    "lassie":   "girl",
    "laddie":   "boy",
    "auld":     "old",
    "cauld":    "cold",
    "heid":     "head",
    "lugs":     "ears",
    "bide":     "live/stay",
    "stey":     "stay/steep",
    "haud":     "hold",
    "tak":      "take",
    "gie":      "give",
    "dae":      "do",
    "hae":      "have",
    "mair":     "more",
    "muckle":   "large/much",
    "twa":      "two",
    "wisnae":   "wasn't",
    "isnae":    "isn't",
    "cannae":   "can't",
    "dinnae":   "don't",
    "didnae":   "didn't",
    "hasnae":   "hasn't",
    "wouldnae": "wouldn't",
    "couldnae": "couldn't",
    "polis":    "police",
    "greet":    "cry",
    "scunnered": "fed up",
    "drookit":  "soaked",
    "stooshie": "commotion/fuss",
    "blether":  "chat/talk nonsense",
    "scran":    "food",
    "gallus":   "bold/cheeky",
    "glaikit":  "foolish-looking",
    "shoogly":  "unstable/wobbly",
    "outwith":  "outside of",
}


def _metaphone_similarity(word1: str, word2: str) -> float:
    """
    Phonetic similarity between two words using Double Metaphone codes
    compared with Jaro-Winkler similarity. Returns 0-1, higher = more similar.
    """
    try:
        m1 = jellyfish.metaphone(word1)
        m2 = jellyfish.metaphone(word2)
        if not m1 or not m2:
            return 0.0
        return jellyfish.jaro_winkler_similarity(m1, m2)
    except Exception:
        return 0.0


def find_phonetic_candidates(
    transcript: str,
    threshold: float = PHONETIC_THRESHOLD,
) -> List[dict]:
    """
    Scan transcript for words phonetically similar to Scottish lexicon entries.

    Returns list of candidate dicts:
        {word, position, scottish_match, meaning, similarity, context}
    """
    words = transcript.split()
    candidates = []

    for i, word in enumerate(words):
        clean = word.strip(".,!?;:'\"").lower()
        if len(clean) < MIN_WORD_LENGTH:
            continue

        # skip if word IS already a Scottish lexicon word
        if clean in SCOTTISH_LEXICON:
            continue

        # skip high-frequency common English words — coincidental matches only
        if clean in COMMON_WORD_STOPLIST:
            continue

        best_match = None
        best_score = 0.0

        for scottish_word in SCOTTISH_LEXICON:
            score = _metaphone_similarity(clean, scottish_word)
            if score > best_score:
                best_score  = score
                best_match  = scottish_word

        if best_match and best_score >= threshold:
            ctx_start = max(0, i - 4)
            ctx_end   = min(len(words), i + 5)
            context   = " ".join(words[ctx_start:ctx_end])

            candidates.append({
                "word":           word,
                "position":       i,
                "scottish_match": best_match,
                "meaning":        SCOTTISH_LEXICON[best_match],
                "similarity":     round(best_score, 3),
                "context":        context,
            })

    return candidates


# ── Semantic judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are checking ONE word in an ASR transcript for a possible mis-transcription of a Scottish dialect word.

Sentence context: "...{context}..."

The ASR transcribed the word as: "{original_word}"
A phonetically similar Scottish word is: "{scottish_word}" (meaning: {meaning})

DEFAULT ASSUMPTION: "{original_word}" is almost certainly correct as transcribed. ASR models are generally accurate on common English words. A phonetic match to a Scottish word is usually coincidental, NOT evidence of an error.

Only answer YES if ALL of the following are true:
1. "{original_word}" creates an actual semantic problem in the sentence as currently written (confusing, contradictory, or nonsensical)
2. "{scottish_word}" resolves that specific problem
3. You would bet money the transcription is wrong, not just "could possibly be" wrong

If "{original_word}" makes ordinary, sensible sense in the sentence — even if "{scottish_word}" would ALSO make sense — answer NO. Plausibility of the Scottish word alone is NOT sufficient grounds for a change.

Reply with ONLY "YES" or "NO". No explanation."""


def judge_candidate(
    candidate: dict,
    client: Client,
    model: str = JUDGE_MODEL,
) -> bool:
    """
    Ask LLM whether the Scottish candidate should replace the original word.
    Conservative — defaults to False (no change) on any uncertainty or error.
    """
    prompt = JUDGE_PROMPT.format(
        context=candidate["context"],
        original_word=candidate["word"],
        scottish_word=candidate["scottish_match"],
        meaning=candidate["meaning"],
    )

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 512},
        )
        answer = response.message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"  ERROR (dialect judge): {e}")
        return False  # conservative default


# ── Main entry point ─────────────────────────────────────────────────────────

def run_dialect_pass(
    transcript: str,
    client: Client,
    model: str = JUDGE_MODEL,
    phonetic_threshold: float = PHONETIC_THRESHOLD,
    sleep: float = 0.05,
) -> Tuple[str, dict]:
    """
    Run the full dialect correction pass on a transcript.

    1. Phonetic scan for Scottish lexicon candidates
    2. Semantic judge for each candidate (conservative — defaults to no change)
    3. Apply approved substitutions

    Args:
        transcript: the transcript to check (post-ROVER, post-auditor)
        client:     Ollama client
        model:      Ollama model name
        phonetic_threshold: minimum similarity to flag a candidate
        sleep:      sleep between Ollama calls

    Returns:
        (corrected_transcript, log_dict)
    """
    candidates = find_phonetic_candidates(transcript, phonetic_threshold)

    if not candidates:
        return transcript, {
            "n_candidates": 0,
            "n_approved":   0,
            "substitutions": [],
        }

    words = transcript.split()
    substitutions = []

    for cand in candidates:
        approved = judge_candidate(cand, client, model)
        time.sleep(sleep)

        substitutions.append({
            **cand,
            "approved": approved,
        })

        if approved:
            pos = cand["position"]
            # preserve original capitalisation pattern roughly
            original = words[pos]
            replacement = cand["scottish_match"]
            if original[0].isupper():
                replacement = replacement.capitalize()
            # preserve trailing punctuation
            trailing = ""
            for ch in reversed(original):
                if ch in ".,!?;:":
                    trailing = ch + trailing
                else:
                    break
            words[pos] = replacement + trailing

            print(f"  [dialect] '{cand['word']}' → '{replacement}' "
                  f"(similarity={cand['similarity']}, context: {cand['context']})")

    corrected = " ".join(words)
    n_approved = sum(1 for s in substitutions if s["approved"])

    return corrected, {
        "n_candidates":  len(candidates),
        "n_approved":    n_approved,
        "substitutions": substitutions,
    }


def check_dialect_pass_available(client: Client, model: str = JUDGE_MODEL) -> bool:
    try:
        available = [m.model for m in client.list().models]
        return any(model in m for m in available)
    except Exception:
        return False