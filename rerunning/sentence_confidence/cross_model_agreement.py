"""
rerunning/sentence_confidence/cross_model_agreement.py

Method 3: Cross-model agreement confidence.

THIRD FIX (review caught this before freezing): segments with
severity=None (alignment_failed) were `continue`-ing BEFORE the cursor
update, meaning the cursor never advanced past skipped segments - the
next SCORED segment then had to search through that skipped content,
reopening the exact drift risk the cursor fix was meant to close.
FIXED: every segment (scored or not) now updates the cursor; only the
save/evaluate step is conditional on having a severity label.

FOURTH FIX: the early-stop condition compared `start` against a fixed
offset from the ORIGINAL cursor (`cursor + sent_len + 5`), not from
where the best match was actually found - so even after finding a
near-perfect match early, the search kept going for a while, risking a
later coincidental repeat outscoring the correct nearby match. Now
compares against `best_start + 5` instead, correctly tying the stop
condition to "searched sufficiently past my best candidate."

RENAMED: per_model_coverage -> per_model_alignment_score. The stored
value is a SequenceMatcher similarity ratio, not a fraction-matched
coverage metric - the old name was inaccurate for the methods section.

METHODOLOGICAL FRAMING (still true, more precisely stated): the fused
segment is used only to LOCATE the corresponding region within each
source hypothesis - it still drives WHERE to look, so this isn't fully
independent of the anchor. What removes the WORST circularity is that,
once the region is chosen, the complete matched span is retained
INCLUDING disagreeing words, rather than only the words matching the
anchor - so agreement is calculated between source-model
reconstructions directly, not filtered through anchor overlap.

No LLM calls - pure text comparison, independent of anything using
Ollama.

Usage:
    python cross_model_agreement.py --dataset commonvoice --split test
"""

import json
import os
import re
import difflib
import argparse
from itertools import combinations

from src.selector import find_canonical_file, load_samples

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
OUTPUT_DIR = "writeup_results/sentence_confidence/cross_model_agreement"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2", "whisper_ft_chunked"]
MAX_EXTRA = 10


def load_segments(dataset, split):
    path = os.path.join(SEGMENTS_DIR, f"sentence_segments_{dataset}_{split}.json")
    return json.load(open(path))


def load_model_words(dataset, model):
    """Returns {dataset_index: [word, word, ...]} - plain word lists
    from each source model's full hyp text (lowercased + whitespace
    tokenized at load time, so case is already normalized uniformly
    before anything downstream touches it)."""
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    result = {}
    for s in samples:
        idx = s.get("sample_index")
        if idx is None or not s.get("hyp"):
            continue
        result[idx] = re.findall(r"[\w']+", s["hyp"].lower())
    return result


def find_local_reconstruction(segment_text, word_list, cursor):
    """Returns (local_words, alignment_score, new_cursor). Scores
    candidate windows via difflib.SequenceMatcher whole-span
    similarity. Cursor persists forward across sequential calls within
    the same transcript - EVERY segment must call this and advance the
    cursor, scored or not (see run_dataset), or the drift bug this
    fixes reopens for the next scored segment.

    Early-stop now compares against best_start (where the best match
    so far was actually found), not the original cursor - stops once
    we've searched sufficiently past our best candidate, not just past
    a fixed offset from where we started looking."""
    anchor_tokens = re.findall(r"[\w']+", segment_text.lower())
    if not anchor_tokens:
        return None, 0.0, cursor

    sent_len = len(anchor_tokens)
    min_len, max_len = max(1, sent_len - MAX_EXTRA), sent_len + MAX_EXTRA
    search_end = min(len(word_list), cursor + sent_len * 3 + MAX_EXTRA)

    best_start, best_end, best_score = None, None, 0.0
    for start in range(cursor, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(word_list):
                continue
            score = difflib.SequenceMatcher(None, anchor_tokens, word_list[start:end]).ratio()
            if score > best_score:
                best_score, best_start, best_end = score, start, end
        if best_start is not None and best_score >= 0.88 and start > best_start + 5:
            break

    if best_start is None:
        return None, 0.0, cursor

    local_words = word_list[best_start:best_end]
    return local_words, best_score, best_end


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
    substitutions, deletions, and insertions - unlike Jaccard. Words
    are already lowercased at load time (load_model_words), so no
    case-normalization needed here."""
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
        cursors_by_model = {m: 0 for m in model_words}

        out_segments = []
        for seg in t.get("segments", []):
            # ALWAYS reconstruct/advance cursors for EVERY segment,
            # scored or not - only the save step below is conditional
            # on having a severity label. Skipping the cursor update
            # for unscored segments reopens the drift bug for the next
            # scored segment.
            per_model_reconstruction = {}
            per_model_alignment_score = {}
            for model, word_data in model_words.items():
                word_list = word_data.get(idx, [])
                cursor = cursors_by_model[model]
                local, score, new_cursor = find_local_reconstruction(seg["segment"], word_list, cursor)
                cursors_by_model[model] = new_cursor
                per_model_alignment_score[model] = score
                if local:
                    per_model_reconstruction[model] = local

            if seg.get("severity") is None:
                continue

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
                "per_model_alignment_score": per_model_alignment_score,
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