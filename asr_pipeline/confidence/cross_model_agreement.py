"""Four-source sentence confidence: cross_model_agreement."""

from asr_pipeline.confidence.paths import SentenceConfidenceFiles

import json
import os
import re
import difflib
import argparse
from itertools import combinations

from asr_pipeline.confidence.inputs import (
    SOURCE_MODELS,
    load_source_hypotheses,
    load_confidence_segments,
)

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
ASR_MODELS = list(SOURCE_MODELS)
MAX_EXTRA = 10


def load_segments(dataset, split, paths=None):
    paths = paths or SentenceConfidenceFiles()
    return load_confidence_segments(dataset, split, paths=paths)


def load_model_words(dataset, model, split, *, source_hypotheses=None, paths=None):
    """Returns {dataset_index: [word, word, ...]} - plain word lists
    from each source model's full hyp text (lowercased + whitespace
    tokenized at load time, so case is already normalized uniformly
    before anything downstream touches it)."""
    paths = paths or SentenceConfidenceFiles()
    if model not in SOURCE_MODELS:
        raise ValueError(f"Not a main four-source ASR model: {model}")
    sources = source_hypotheses
    if sources is None:
        sources = load_source_hypotheses(dataset, split, paths=paths)
    result = {}
    for transcript_index, hypotheses in sources.items():
        if not hypotheses[model]:
            continue
        result[transcript_index] = re.findall(r"[\w']+", hypotheses[model].lower())
    return result


def find_local_reconstruction(segment_text, word_list, cursor):
    """Find the source words that best match a sentence.

    Search forward from cursor with SequenceMatcher. Return the words, match score
    and next cursor. Move the cursor for every sentence, even one without a label,
    so later sentences stay aligned."""
    anchor_tokens = re.findall(r"[\w']+", segment_text.lower())
    if not anchor_tokens:
        return None, 0.0, cursor

    sentence_word_count = len(anchor_tokens)
    min_len, max_len = (
        max(1, sentence_word_count - MAX_EXTRA),
        sentence_word_count + MAX_EXTRA,
    )
    search_end = min(len(word_list), cursor + sentence_word_count * 3 + MAX_EXTRA)

    best_start, best_end, best_score = None, None, 0.0
    for start in range(cursor, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(word_list):
                continue
            score = difflib.SequenceMatcher(
                None, anchor_tokens, word_list[start:end]
            ).ratio()
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
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution / match
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


def run_dataset(
    dataset,
    split="test",
    *,
    segments_data=None,
    source_hypotheses=None,
    output_dir=None,
    include_unlabelled=False,
    paths=None,
):
    paths = paths or SentenceConfidenceFiles()
    print(f"\n-- {dataset} | cross_model_agreement split={split} --")

    if segments_data is None:
        segments_data = load_segments(dataset, split, paths=paths)
    transcripts = segments_data.get("transcripts", [])

    model_words = {}
    for model in ASR_MODELS:
        model_words[model] = load_model_words(
            dataset, model, split, source_hypotheses=source_hypotheses, paths=paths
        )
        print(f"  loaded word data for {model}")

    results = []
    for t in transcripts:
        transcript_index = t["dataset_index"]
        cursors_by_model = {m: 0 for m in model_words}

        out_segments = []
        for segment in t.get("segments", []):
            # ALWAYS reconstruct/advance cursors for EVERY segment,
            # scored or not - only the save step below is conditional
            # on having a severity label. Skipping the cursor update
            # for unscored segments reopens the drift bug for the next
            # scored segment.
            per_model_reconstruction = {}
            per_model_alignment_score = {}
            for model, word_data in model_words.items():
                word_list = word_data.get(transcript_index, [])
                cursor = cursors_by_model[model]
                local, score, new_cursor = find_local_reconstruction(
                    segment["segment"], word_list, cursor
                )
                cursors_by_model[model] = new_cursor
                per_model_alignment_score[model] = score
                if local:
                    per_model_reconstruction[model] = local

            if segment.get("severity") is None and not include_unlabelled:
                continue

            pairwise_scores = []
            for model_a, model_b in combinations(per_model_reconstruction.keys(), 2):
                score = edit_distance_agreement(
                    per_model_reconstruction[model_a], per_model_reconstruction[model_b]
                )
                pairwise_scores.append(score)

            crossmodel_mean = (
                sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else None
            )
            crossmodel_min = min(pairwise_scores) if pairwise_scores else None

            out_segments.append(
                {
                    "segment": segment["segment"],
                    "severity": segment.get("severity"),
                    "flagged": segment.get("flagged"),
                    "n_models_reconstructed": len(per_model_reconstruction),
                    "local_reconstructions": {
                        m: " ".join(words)
                        for m, words in per_model_reconstruction.items()
                    },
                    "per_model_alignment_score": per_model_alignment_score,
                    "crossmodel_mean": crossmodel_mean,
                    "crossmodel_min": crossmodel_min,
                }
            )

        results.append({"dataset_index": transcript_index, "segments": out_segments})

    output_dir = output_dir or paths.agreement_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"cross_model_agreement_{dataset}_{split}.json"
    )

    segment_count = sum(len(t["segments"]) for t in results)
    scored_count = sum(
        1
        for t in results
        for s in t["segments"]
        if s.get("crossmodel_mean") is not None
    )

    output = {
        "dataset": dataset,
        "split": split,
        "method": "cross_model_agreement",
        "agreement_metric": "normalized_word_level_edit_distance",
        "asr_models_used": list(model_words.keys()),
        "num_transcripts": len(results),
        "num_segments": segment_count,
        "num_segments_scored": scored_count,
        "transcripts": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  {scored_count}/{segment_count} segments scored. Saved: {output_path}")
    return output


def main(argv=None, paths=None):
    paths = paths or SentenceConfidenceFiles()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=[*DATASETS, "clip"])
    parser.add_argument("--split", default="test", choices=["dev", "test", "full"])
    parser.add_argument(
        "--include-unlabelled",
        action="store_true",
        help="also score sentences that have no evaluation labels",
    )
    args = parser.parse_args(argv)

    run_dataset(
        args.dataset,
        split=args.split,
        paths=paths,
        include_unlabelled=args.include_unlabelled,
    )


if __name__ == "__main__":
    main()
