"""
src/segmenter.py

Sentence-level segmentation for multi-model ASR combination.

Uses spaCy to split Qwen's transcript into sentences, then aligns
each other model's output against Qwen word-by-word to find where
sentence boundaries fall in their sequences.

Output: list of sentence-level hypothesis tuples, one per sentence,
ready to feed into segment-level ROVER.

Usage:
    from src.segmenter import segment_hypotheses

    sentences = segment_hypotheses({
        "qwen":     "I think so. She was there.",
        "whisper":  "I think so She wisnae there",
        "parakeet": "i think so she was there",
        "wav2vec2": "I THINK SO SHE WAS THERE",
    })
    # returns list of dicts, one per sentence:
    # [
    #   {"qwen": "I think so", "whisper": "I think so", ...},
    #   {"qwen": "She was there", "whisper": "She wisnae there", ...},
    # ]
"""

import re
from typing import Dict, List, Optional

# ── spaCy loader ───────────────────────────────────────────────────────────────

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # use sentencizer only — faster, no need for full NLP pipeline
            try:
                _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
            except OSError:
                # fallback: blank English with sentencizer
                _nlp = spacy.blank("en")
                _nlp.add_pipe("sentencizer")
        except ImportError:
            _nlp = None
    return _nlp


# ── Sentence splitting ─────────────────────────────────────────────────────────

def _split_sentences_spacy(text: str) -> List[str]:
    """Split text into sentences using spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        # fallback: split on sentence-ending punctuation
        return _split_sentences_fallback(text)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text]


def _split_sentences_fallback(text: str) -> List[str]:
    """Fallback sentence splitter using punctuation."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()] or [text]


def _tokenise(text: str) -> List[str]:
    """Simple whitespace tokenisation."""
    return text.strip().split()


# ── Word-level alignment ───────────────────────────────────────────────────────

def _align_to_reference(
    ref_words: List[str],
    hyp_words: List[str],
) -> List[Optional[int]]:
    """
    Align hyp_words against ref_words using DP.
    Returns a list of length len(ref_words) where each entry is the index
    in hyp_words that corresponds to that reference word, or None if deleted.

    This tells us: "ref word i corresponds to hyp word alignment[i]"
    so we can find sentence boundaries in hyp by looking at where the
    boundary falls in ref and mapping through alignment.
    """
    m, n = len(ref_words), len(hyp_words)

    def norm(w):
        return w.lower().strip(".,!?;:")

    # DP cost matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = norm(ref_words[i-1]) == norm(hyp_words[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,                    # delete from ref
                dp[i][j-1] + 1,                    # insert in hyp
                dp[i-1][j-1] + (0 if match else 1) # match/sub
            )

    # traceback — build alignment: ref_idx -> hyp_idx or None
    alignment = [None] * m
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = norm(ref_words[i-1]) == norm(hyp_words[j-1])
            cost_diag = dp[i-1][j-1] + (0 if match else 1)
            cost_del  = dp[i-1][j] + 1
            cost_ins  = dp[i][j-1] + 1

            if dp[i][j] == cost_diag:
                alignment[i-1] = j-1  # ref[i-1] aligns to hyp[j-1]
                i -= 1; j -= 1
            elif dp[i][j] == cost_del:
                alignment[i-1] = None  # ref word deleted in hyp
                i -= 1
            else:
                j -= 1  # hyp insertion — skip
        elif i > 0:
            alignment[i-1] = None
            i -= 1
        else:
            j -= 1

    return alignment


def _ends_with_punctuation(word: str) -> bool:
    """True if word ends with sentence-terminating punctuation."""
    return bool(word) and word.rstrip()[-1] in ".!?"


def _snap_to_punctuation(
    hyp_words: List[str],
    candidate_end: int,
    search_window: int = 4,
) -> int:
    """
    Given a candidate boundary position in hyp_words, search ±window
    words for a word ending in sentence punctuation. If found, snap
    the boundary there. Otherwise return candidate unchanged.
    """
    lo = max(0, candidate_end - search_window)
    hi = min(len(hyp_words), candidate_end + search_window + 1)

    # prefer punctuation at or just before the candidate (within window)
    # search backward from candidate first, then forward
    for offset in range(0, search_window + 1):
        for direction in [-1, 1]:
            pos = candidate_end + direction * offset - 1  # -1: word before pos
            if lo <= pos < hi and _ends_with_punctuation(hyp_words[pos]):
                return pos + 1  # boundary = after this word
    return candidate_end


def _split_hyp_by_ref_boundaries(
    ref_words: List[str],
    hyp_words: List[str],
    boundary_positions: List[int],  # positions in ref where sentences end
) -> List[str]:
    """
    Given sentence boundary positions in ref, find corresponding positions
    in hyp via alignment, and split hyp into matching segments.

    Uses punctuation snapping: after DP alignment gives a candidate boundary,
    we search ±4 words for sentence-terminating punctuation and snap there
    if found. This prevents boundary bleed when models have minor insertions
    or deletions relative to the anchor.

    boundary_positions: list of end indices (exclusive) in ref_words
    e.g. [3, 7] means sentence 1 = ref[0:3], sentence 2 = ref[3:7]
    """
    if not hyp_words:
        return [""] * len(boundary_positions)

    alignment = _align_to_reference(ref_words, hyp_words)

    segments    = []
    prev_hyp_end = 0

    for i, ref_end in enumerate(boundary_positions):
        is_last = (i == len(boundary_positions) - 1)

        if is_last:
            segment_words = hyp_words[prev_hyp_end:]
        else:
            # 1. find DP-aligned candidate boundary
            hyp_end = None
            for ref_idx in range(ref_end - 1, -1, -1):
                if alignment[ref_idx] is not None:
                    hyp_end = alignment[ref_idx] + 1
                    break

            if hyp_end is None or hyp_end <= prev_hyp_end:
                # fallback: proportional split
                fraction = ref_end / max(len(ref_words), 1)
                hyp_end  = max(prev_hyp_end + 1,
                               int(fraction * len(hyp_words)))

            hyp_end = min(hyp_end, len(hyp_words))

            # 2. snap to nearby punctuation if available
            hyp_end = _snap_to_punctuation(
                hyp_words, hyp_end, search_window=4
            )
            hyp_end = max(prev_hyp_end + 1, min(hyp_end, len(hyp_words)))

            segment_words = hyp_words[prev_hyp_end:hyp_end]
            prev_hyp_end  = hyp_end

        segments.append(" ".join(segment_words).strip())

    return segments


# ── Main segmentation function ─────────────────────────────────────────────────

def segment_hypotheses(
    hypotheses: Dict[str, str],
    anchor_model: str = "qwen",
) -> List[Dict[str, str]]:
    """
    Segment multi-model ASR hypotheses into sentence-level chunks.

    Args:
        hypotheses: dict of {model_name: transcript_text}
        anchor_model: model to use for sentence boundary detection (default: qwen)

    Returns:
        List of dicts, one per sentence:
        [{"qwen": "...", "whisper": "...", ...}, ...]

    If anchor model not in hypotheses, falls back to first available model.
    """
    if not hypotheses:
        return []

    # pick anchor
    anchor = anchor_model if anchor_model in hypotheses else list(hypotheses.keys())[0]
    anchor_text = hypotheses[anchor]

    # split anchor into sentences
    sentences = _split_sentences_spacy(anchor_text)

    if len(sentences) <= 1:
        # single sentence — no splitting needed
        return [hypotheses.copy()]

    # find boundary positions in anchor word sequence
    anchor_words = _tokenise(anchor_text)
    boundary_positions = []
    word_count = 0
    sent_word_lists = [_tokenise(s) for s in sentences]

    for sent_words in sent_word_lists:
        word_count += len(sent_words)
        boundary_positions.append(word_count)

    # ensure last boundary covers all words
    boundary_positions[-1] = len(anchor_words)

    # build output: one dict per sentence
    result = []
    for i in range(len(sentences)):
        seg = {}
        for model, text in hypotheses.items():
            if model == anchor:
                seg[model] = sentences[i]
            else:
                hyp_words = _tokenise(text)
                segs = _split_hyp_by_ref_boundaries(
                    anchor_words, hyp_words, boundary_positions
                )
                seg[model] = segs[i] if i < len(segs) else ""
        result.append(seg)

    return result


def segment_with_indices(
    hypotheses: Dict[str, str],
    anchor_model: str = "qwen",
) -> List[Dict]:
    """
    Like segment_hypotheses but also returns sentence index and
    original anchor sentence for reference.
    """
    segments = segment_hypotheses(hypotheses, anchor_model)
    return [
        {"sentence_idx": i, "hypotheses": seg}
        for i, seg in enumerate(segments)
    ]