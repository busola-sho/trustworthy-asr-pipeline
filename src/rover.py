"""
src/rover.py

Pure-Python ROVER implementation for ASR system combination.

Implements:
1. Iterative pairwise word alignment using dynamic programming
2. Confusion network construction
3. Weighted voting at each word position
4. Word-level confidence scores (vote_count / n_models)
5. Disagreement detection and classification

Based on: Fiscus (1997) "A post-processing system to yield reduced word
error rates: Recognizer Output Voting Error Reduction (ROVER)"

Usage:
    from src.rover import rover_combine, classify_disagreements

    result = rover_combine([
        ("qwen",     "I think there's damage done on the internet"),
        ("whisper",  "I think there's damage being done on the internet"),
        ("parakeet", "i think there's damage been done on the internet"),
        ("wav2vec2", "I THINK THERES DAMAGE DONE ON THE INTERNET"),
    ])

    print(result.transcript)        # best combined transcript
    print(result.word_confidences)  # per-word confidence scores
    print(result.utterance_confidence)  # single utterance score
    print(result.disagreements)     # list of disagreement spans
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from src.judge import normalise

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class WordHyp:
    """A word hypothesis at one position in the confusion network."""
    word: str           # surface form
    model: str          # which model produced it
    confidence: float   # model's word-level confidence (if available)
    is_null: bool = False  # True = deletion (epsilon arc)


@dataclass
class ConfusionSlot:
    """One position in the confusion network — a set of competing hypotheses."""
    hypotheses: List[WordHyp]

    @property
    def total_votes(self):
        return len(self.hypotheses)

    @property
    def n_null(self):
        return sum(1 for h in self.hypotheses if h.is_null)

    def vote(self, weights: Optional[Dict[str, float]] = None):
        """
        Return (best_word, confidence) by weighted voting.
        confidence = weighted_votes_for_winner / total_weight
        """
        if weights is None:
            weights = {h.model: 1.0 for h in self.hypotheses}

        counts: Dict[str, float] = {}
        total_weight = sum(weights.get(h.model, 1.0) for h in self.hypotheses)

        for h in self.hypotheses:
            w = weights.get(h.model, 1.0)
            key = "" if h.is_null else normalise(h.word)
            counts[key] = counts.get(key, 0.0) + w

        # pick winner (highest weighted votes, "" = deletion)
        best_key = max(counts, key=lambda k: counts[k])
        confidence = counts[best_key] / total_weight if total_weight > 0 else 0.0

        # recover original surface form from winner
        if best_key == "":
            return None, confidence  # deletion wins
        for h in self.hypotheses:
            if not h.is_null and normalise(h.word) == best_key:
                return h.word, confidence
        return best_key, confidence


@dataclass
class ROVERResult:
    """Output of ROVER combination."""
    transcript: str
    word_confidences: List[Tuple[str, float]]  # [(word, confidence), ...]
    utterance_confidence: float
    confusion_network: List[ConfusionSlot]
    disagreements: List[dict]  # list of disagreement spans


# ── DP word alignment ──────────────────────────────────────────────────────────

def _align_sequences(
    seq_a: List[WordHyp],
    seq_b: List[WordHyp],
    ins_cost: float = 1.0,
    del_cost: float = 1.0,
    sub_cost: float = 1.0,
) -> List[ConfusionSlot]:
    """
    Align two sequences of WordHyp using DP (Levenshtein).
    Returns a list of ConfusionSlots representing the aligned confusion network.
    """
    m, n = len(seq_a), len(seq_b)

    # DP cost matrix
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * del_cost
    for j in range(n + 1):
        dp[0][j] = j * ins_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = normalise(seq_a[i-1].word) == normalise(seq_b[j-1].word)
            dp[i][j] = min(
                dp[i-1][j]   + del_cost,           # delete from a
                dp[i][j-1]   + ins_cost,            # insert from b
                dp[i-1][j-1] + (0.0 if match else sub_cost),  # match/sub
            )

    # traceback
    slots = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = normalise(seq_a[i-1].word) == normalise(seq_b[j-1].word)
            cost_diag = dp[i-1][j-1] + (0.0 if match else sub_cost)
            cost_del  = dp[i-1][j]   + del_cost
            cost_ins  = dp[i][j-1]   + ins_cost

            if dp[i][j] == cost_diag:
                # match or substitution
                slots.append(ConfusionSlot([seq_a[i-1], seq_b[j-1]]))
                i -= 1; j -= 1
            elif dp[i][j] == cost_del:
                null_b = WordHyp(word="", model=seq_b[0].model if seq_b else "?",
                                 confidence=0.0, is_null=True)
                slots.append(ConfusionSlot([seq_a[i-1], null_b]))
                i -= 1
            else:
                null_a = WordHyp(word="", model=seq_a[0].model if seq_a else "?",
                                 confidence=0.0, is_null=True)
                slots.append(ConfusionSlot([null_a, seq_b[j-1]]))
                j -= 1
        elif i > 0:
            null_b = WordHyp(word="", model="?", confidence=0.0, is_null=True)
            slots.append(ConfusionSlot([seq_a[i-1], null_b]))
            i -= 1
        else:
            null_a = WordHyp(word="", model="?", confidence=0.0, is_null=True)
            slots.append(ConfusionSlot([null_a, seq_b[j-1]]))
            j -= 1

    slots.reverse()
    return slots


def _merge_slot_with_hyp(slot: ConfusionSlot, hyp: WordHyp) -> ConfusionSlot:
    """Add a new hypothesis to an existing confusion slot."""
    return ConfusionSlot(slot.hypotheses + [hyp])


def _align_network_with_sequence(
    network: List[ConfusionSlot],
    seq: List[WordHyp],
) -> List[ConfusionSlot]:
    """
    Align an existing confusion network with a new sequence.
    Treats the network's best-path (majority vote) as the reference for alignment.
    """
    # get best path from current network for alignment reference
    best_path = []
    for slot in network:
        word, _ = slot.vote()
        if word:
            best_path.append(WordHyp(word=word, model="__network__", confidence=1.0))

    # align best path against new sequence
    aligned = _align_sequences(best_path, seq)

    # now merge: map aligned positions back to original network slots
    result = []
    net_idx = 0
    for slot in aligned:
        net_hyps = slot.hypotheses[0]   # from network side
        new_hyp  = slot.hypotheses[1]   # from new sequence

        if net_hyps.is_null:
            # insertion: create new slot with null from all previous models + new word
            null_hyps = [
                WordHyp(word="", model=h.model, confidence=0.0, is_null=True)
                for h in network[0].hypotheses
            ] if network else []
            result.append(ConfusionSlot(null_hyps + [new_hyp]))
        else:
            if net_idx < len(network):
                merged = _merge_slot_with_hyp(network[net_idx], new_hyp)
                result.append(merged)
                net_idx += 1

    return result


# ── Disagreement classification ────────────────────────────────────────────────

# High-risk categories for meaning alteration
SCOTTISH_NEGATION_WORDS = {
    "wisnae", "isnae", "isna", "cannae", "canna", "dinnae", "dinna",
    "didnae", "didna", "hasnae", "hasna", "havnae", "wouldnae", "couldnae",
    "shouldnae", "willnae", "wasnae", "nae", "naw",
}

STANDARD_NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "nobody", "nothing",
    "nowhere", "wasn't", "isn't", "aren't", "weren't", "haven't", "hasn't",
    "hadn't", "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't",
    "didn't", "doesn't", "don't",
}

NUMBER_PATTERN = re.compile(r'\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|'
                             r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
                             r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
                             r'eighty|ninety|hundred|thousand|million)\b', re.IGNORECASE)


def classify_disagreement(slot: ConfusionSlot, context_words: List[str]) -> str:
    """
    Classify a disagreement slot as one of:
    - 'easy': 3+ models agree (majority is clear)
    - 'negation': involves a Scottish negation word (wisnae, cannae etc.)
                  OR a standard negation word where models genuinely split
    - 'number': involves a number or quantity
    - 'named_entity': likely a proper noun
    - 'ambiguous': unclear
    """
    words = [h.word for h in slot.hypotheses if not h.is_null]
    normalised = [normalise(w) for w in words]

    from collections import Counter
    counts    = Counter(normalised)
    top_count = counts.most_common(1)[0][1] if counts else 0

    all_words_in_slot = set(normalised)

    # Scottish negation: flag immediately regardless of majority
    # e.g. "was" vs "wisnae" — minority model may be correct
    if any(w in SCOTTISH_NEGATION_WORDS for w in all_words_in_slot):
        return "negation"

    # easy majority (checked before standard negation)
    if top_count >= 3:
        return "easy"

    # standard negation: only flag if models genuinely split
    # (not covered by easy majority above)
    if any(w in STANDARD_NEGATION_WORDS for w in all_words_in_slot):
        return "negation"

    # check for numbers
    if any(NUMBER_PATTERN.search(w) for w in words if w):
        return "number"

    # check for likely named entity (capitalised but not all-caps wav2vec2 artifact)
    non_null = [w for w in words if w]
    if non_null and any(w[0].isupper() and not w.isupper() for w in non_null if w):
        return "named_entity"

    return "ambiguous"


# ── Main ROVER combination ─────────────────────────────────────────────────────

def rover_combine(
    hypotheses: List[Tuple[str, str]],
    weights: Optional[Dict[str, float]] = None,
    word_confidences: Optional[Dict[str, List[Tuple[str, float]]]] = None,
) -> ROVERResult:
    """
    Combine N ASR hypotheses using ROVER.

    Uses the first (highest-confidence) model as a fixed anchor.
    All other models are aligned against this anchor independently,
    then merged into a confusion network position by position.

    Args:
        hypotheses: List of (model_name, transcript_text) tuples,
                    ordered highest-confidence first
        weights: Optional per-model weights for voting
        word_confidences: Optional {model: [(word, conf), ...]}

    Returns:
        ROVERResult with transcript, word confidences, and disagreements
    """
    if not hypotheses:
        return ROVERResult("", [], 0.0, [], [])

    # tokenise each hypothesis
    sequences = []
    for model, text in hypotheses:
        words = text.strip().split()
        model_confs = dict(word_confidences.get(model, [])) if word_confidences else {}
        seqs = [
            WordHyp(
                word=w,
                model=model,
                confidence=model_confs.get(w, 0.8),
            )
            for w in words
        ]
        sequences.append((model, seqs))

    # use first model as fixed anchor
    anchor_model, anchor_seq = sequences[0]

    # build confusion network: one slot per anchor word initially
    # each slot starts with just the anchor word
    network: List[ConfusionSlot] = [
        ConfusionSlot([hyp]) for hyp in anchor_seq
    ]

    # align each subsequent model against the fixed anchor
    for model, seq in sequences[1:]:
        aligned = _align_sequences(anchor_seq, seq)
        # aligned is a list of ConfusionSlots with 2 hypotheses each:
        # [anchor_hyp, other_hyp]
        # map back to network positions using anchor_seq indices

        # build a mapping: anchor_word_idx -> list of other_hyps aligned to it
        anchor_to_other: Dict[int, List[WordHyp]] = {i: [] for i in range(len(anchor_seq))}
        insertions_before: Dict[int, List[WordHyp]] = {i: [] for i in range(len(anchor_seq) + 1)}

        anchor_pos = 0
        for slot in aligned:
            a_hyp = slot.hypotheses[0]  # anchor side
            o_hyp = slot.hypotheses[1]  # other model side

            if a_hyp.is_null:
                # insertion in other — attach before current anchor position
                insertions_before[anchor_pos].append(o_hyp)
            else:
                # match or substitution — attach to current anchor position
                anchor_to_other[anchor_pos].append(o_hyp)
                anchor_pos += 1

        # rebuild network incorporating new model
        new_network: List[ConfusionSlot] = []
        for i, slot in enumerate(network):
            # add any insertions before this position
            for ins_hyp in insertions_before.get(i, []):
                null_hyps = [
                    WordHyp(word="", model=h.model, confidence=0.0, is_null=True)
                    for h in slot.hypotheses
                ]
                new_network.append(ConfusionSlot(null_hyps + [ins_hyp]))

            # add the aligned other hypotheses for this anchor position
            other_hyps = anchor_to_other.get(i, [])
            if other_hyps:
                merged = ConfusionSlot(slot.hypotheses + other_hyps)
            else:
                # other model deleted this position — add null
                null = WordHyp(word="", model=model, confidence=0.0, is_null=True)
                merged = ConfusionSlot(slot.hypotheses + [null])
            new_network.append(merged)

        # add any trailing insertions after last anchor position
        for ins_hyp in insertions_before.get(len(anchor_seq), []):
            null_hyps = [
                WordHyp(word="", model=h.model, confidence=0.0, is_null=True)
                for h in (network[-1].hypotheses if network else [])
            ]
            new_network.append(ConfusionSlot(null_hyps + [ins_hyp]))

        network = new_network

    # vote at each slot
    output_words = []
    word_confs   = []

    for slot in network:
        word, conf = slot.vote(weights=weights)
        if word:
            output_words.append(word)
            word_confs.append((word, conf))

    # detect disagreements
    disagreements = []
    output_pos = 0
    for pos, slot in enumerate(network):
        words_in_slot = set(
            normalise(h.word) for h in slot.hypotheses if not h.is_null
        )
        null_count = sum(1 for h in slot.hypotheses if h.is_null)

        word, conf = slot.vote(weights=weights)
        if word:
            output_pos += 1

        if len(words_in_slot) > 1 or null_count > 0:
            category = classify_disagreement(slot, [])
            disagreements.append({
                "position":   pos,
                "output_pos": output_pos - 1 if word else output_pos,
                "category":   category,
                "best_word":  word,
                "confidence": conf,
                "hypotheses": {
                    h.model: h.word for h in slot.hypotheses
                },
            })

    transcript     = " ".join(output_words)
    utterance_conf = (
        sum(c for _, c in word_confs) / len(word_confs)
        if word_confs else 0.0
    )

    return ROVERResult(
        transcript=transcript,
        word_confidences=word_confs,
        utterance_confidence=utterance_conf,
        confusion_network=network,
        disagreements=disagreements,
    )


def risky_disagreements(result: ROVERResult) -> List[dict]:
    """Return only high-risk disagreements that should go to the LLM.
    Negations are always risky regardless of confidence — a 3/4 majority
    saying "was" when one model says "wisnae" is still a critical ambiguity.
    Other categories require low confidence to be flagged.
    """
    return [
        d for d in result.disagreements
        if d["category"] == "negation"  # always flag negations
        or (
            d["category"] in ("number", "named_entity", "ambiguous")
            and d["confidence"] < 0.75
        )
    ]