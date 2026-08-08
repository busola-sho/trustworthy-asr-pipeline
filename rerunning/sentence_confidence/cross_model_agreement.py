"""
rerunning/sentence_confidence/cross_model_agreement.py

Method 3: Cross-model agreement confidence. Measures how strongly the
ASR hypotheses agree within a given fixed segment - higher agreement
treated as greater confidence, disagreement as uncertainty. This is
an ASR-specific disagreement-based heuristic (not a direct replication
of a single paper).

METHODOLOGICAL FIX (caught by review before running the final
experiments): an earlier version reconstructed each source model's
"local region" by searching for words from the FIXED SEGMENT within
that model's transcript - i.e. "which of the segment's words can I
find in this model's output". This is CIRCULAR: a disagreeing word
(segment says "didn't", model says "did") is never found by that
search, so it silently disappears from the reconstruction instead of
counting as a disagreement - meaning the old design measured "how
much of the segment's own words can be recovered from this model",
not "how much do the models actually agree with each other". It also
used Jaccard word-set overlap, which discards word order and
duplicates entirely (so "man bites dog" and "dog bites man" would
score a perfect 1.0).

FIXED DESIGN: for each source model, find the POSITIONAL RANGE (first
matched word index to last matched word index) that the segment's
words fall within, then extract the ENTIRE local substring across
that range - including any words that did NOT match, which represent
genuine disagreement/different wording, not just the matching subset.
Pairwise agreement between two models' local reconstructions is then
computed via NORMALIZED WORD-LEVEL EDIT DISTANCE:
    agreement(A, B) = 1 - edit_distance(A, B) / max(len(A), len(B))
This retains substitutions, deletions, and insertions, and is
sensitive to word order (standard Levenshtein has no dedicated
transposition/reordering operation - a reordering is scored via the
substitutions/insertions/deletions needed to reach it, not as a
single op - but it still correctly reduces the agreement score,
unlike Jaccard, which ignores order and duplicates entirely). "didn't"
vs "did" or a reordering both correctly reduce the agreement score.

No LLM calls - pure text comparison, independent of anything using
Ollama.

Usage:
    python cross_model_agreement.py --dataset commonvoice --split test
"""

import json
import os
import re
import argparse
from itertools import combinations

from src.selector import find_canonical_file, load_samples

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
OUTPUT_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2", "whisper_ft_chunked"]


def load_segments(dataset, split):
    path = os.path.join(SEGMENTS_DIR, f"sentence_segments_{dataset}_{split}.json")
    return json.load(open(path))


def load_model_words(dataset, model):
    """Returns {dataset_index: [word, word, ...]} - plain word lists
    from each source model's full hyp text (whitespace tokenized)."""
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    result = {}
    for s in samples:
        idx = s.get("sample_index")
        if idx is None or not s.get("hyp"):
            continue
        result[idx] = re.findall(r"[\w']+", s["hyp"].lower())
    return result


def find_local_reconstruction(segment_text, word_list, used_positions):
    """Finds the positional range in word_list that the segment's
    words fall within (via sequential matching, same search pattern as
    before), then returns the FULL substring across that range -
    including any unmatched words within it, which represent genuine
    disagreement rather than being silently dropped. Returns
    (None, 0.0) if nothing in the segment matched at all (no valid
    range to anchor). Marks every position within the returned range
    as used, matched or not, so a later segment from the same
    transcript can't reuse words this segment's range already claimed.

    Also returns match_coverage = n_matched / len(segment_words) - if
    matching is sparse (e.g. only the first and last words of a long
    segment happened to match), the resulting range can be wide and
    pull in a large unrelated span that then reads as "disagreement"
    when it's really just poor localization. Coverage is saved for
    inspection, not filtered on automatically yet."""
    segment_words = re.findall(r"[\w']+", segment_text.lower())
    if not segment_words:
        return None, 0.0

    matched_positions = []
    search_from = 0
    for word in segment_words:
        found_at = None
        for i in range(search_from, len(word_list)):
            if i in used_positions:
                continue
            if word_list[i].strip(".,!?;:\"'") == word:
                found_at = i
                break
        if found_at is not None:
            matched_positions.append(found_at)
            search_from = found_at + 1

    coverage = len(matched_positions) / len(segment_words)

    if not matched_positions:
        return None, coverage

    range_start, range_end = min(matched_positions), max(matched_positions)
    local_words = word_list[range_start:range_end + 1]
    for i in range(range_start, range_end + 1):
        used_positions.add(i)

    return local_words, coverage


def word_edit_distance(a, b):
    """Standard Levenshtein distance at WORD level (not character
    level) - a dynamic-programming edit distance over the two word
    lists, so substitutions/deletions/insertions/reordering at the
    word level are all captured correctly."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution / match
            )
    return dp[n][m]


def edit_distance_agreement(words_a, words_b):
    """Normalized word-level edit-distance agreement: 1.0 = identical
    sequences, 0.0 = maximally different. Retains order, duplicates,
    substitutions, deletions, and insertions - unlike Jaccard."""
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    distance = word_edit_distance(words_a, words_b)
    max_len = max(len(words_a), len(words_b))
    return 1 - (distance / max_len) if max_len > 0 else 1.0


def run_dataset(dataset, split="test"):
    print(f"\n-- {dataset} | cross_model_agreement split={split} --")

    segments_data = load_segments(dataset, split)
    transcripts = segments_data.get("transcripts", [])

    model_words = {}
    for model in ASR_MODELS:
        try:
            model_words[model] = load_model_words(dataset, model)
            print(f"  loaded word data for {model}")
        except FileNotFoundError:
            print(f"  WARNING: no benchmark file found for {model}/{dataset} - skipping this model")

    results = []
    for t in transcripts:
        idx = t["dataset_index"]
        used_positions_by_model = {m: set() for m in model_words}

        out_segments = []
        for seg in t.get("segments", []):
            if seg.get("severity") is None:
                continue

            per_model_reconstruction = {}
            per_model_coverage = {}
            for model, word_data in model_words.items():
                word_list = word_data.get(idx, [])
                local, coverage = find_local_reconstruction(seg["segment"], word_list,
                                                             used_positions_by_model[model])
                per_model_coverage[model] = coverage
                if local:
                    per_model_reconstruction[model] = local

            pairwise_scores = []
            for model_a, model_b in combinations(per_model_reconstruction.keys(), 2):
                score = edit_distance_agreement(per_model_reconstruction[model_a],
                                                per_model_reconstruction[model_b])
                pairwise_scores.append(score)

            crossmodel_mean = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else None
            crossmodel_min = min(pairwise_scores) if pairwise_scores else None

            out_segments.append({
                "segment": seg["segment"],
                "severity": seg.get("severity"),
                "flagged": seg.get("flagged"),
                "n_models_reconstructed": len(per_model_reconstruction),
                "local_reconstructions": {m: " ".join(words) for m, words in per_model_reconstruction.items()},
                "per_model_coverage": per_model_coverage,
                "crossmodel_mean": crossmodel_mean,
                "crossmodel_min": crossmodel_min,
            })

        results.append({"dataset_index": idx, "segments": out_segments})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"cross_model_agreement_{dataset}_{split}.json")

    n_segments = sum(len(t["segments"]) for t in results)
    n_scored = sum(1 for t in results for s in t["segments"] if s.get("crossmodel_mean") is not None)

    output = {
        "dataset": dataset,
        "split": split,
        "method": "cross_model_agreement",
        "agreement_metric": "normalized_word_level_edit_distance",
        "asr_models_used": list(model_words.keys()),
        "num_transcripts": len(results),
        "num_segments": n_segments,
        "num_segments_scored": n_scored,
        "transcripts": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  {n_scored}/{n_segments} segments scored. Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="test", choices=["dev", "test", "full"])
    args = parser.parse_args()

    run_dataset(args.dataset, split=args.split)


if __name__ == "__main__":
    main()
