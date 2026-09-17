"""
rerunning/sentence_confidence/model_internal_confidence.py

Method 2: Model-internal ASR confidence.

Maps native word-level confidence from the source ASR systems onto the
fixed sentence-like segments used throughout the sentence-confidence
experiments.

Key alignment change:
- The previous implementation matched segment words greedily, one word at
  a time, which could drift in the presence of repetitions, omissions, or
  disfluencies.
- This version aligns each sentence-like segment to the best local span in
  each source ASR transcript using whole-span SequenceMatcher similarity.
- The matched span's native word confidences are then averaged to produce
  one raw sentence-level confidence per ASR model.

Normalization:
- Raw confidence scales differ substantially across ASR systems.
- Per-model mean/stdev are therefore fitted once on the pooled development
  data from CommonVoice, EdAcc, and English Dialects.
- These frozen statistics are then applied unchanged to test splits and
  Shetland.

Pipeline:
native word confidence
    -> local sentence-span alignment per model
    -> mean confidence over matched span
    -> per-model z-score normalization
    -> alignment-score-weighted mean across models

Usage:
    python rerunning/sentence_confidence/model_internal_confidence.py --fit-pooled-stats

    python rerunning/sentence_confidence/model_internal_confidence.py \
        --dataset commonvoice --split dev

    python rerunning/sentence_confidence/model_internal_confidence.py \
        --dataset commonvoice --split test

    python rerunning/sentence_confidence/model_internal_confidence.py \
        --dataset shetland --split full
"""

import argparse
import difflib
import json
import os
import re
import statistics

from src.selector import find_canonical_file, load_samples


SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
OUTPUT_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
POOLED_STATS_PATH = os.path.join(OUTPUT_DIR, "norm_stats_pooled.json")

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
IN_DOMAIN_DEV_DATASETS = ["commonvoice", "edacc", "english_dialects"]

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]

MAX_EXTRA = 10


def load_segments(dataset, split):
    path = os.path.join(
        SEGMENTS_DIR,
        f"sentence_segments_{dataset}_{split}.json",
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalise_word(word):
    return re.sub(r"^[^\w']+|[^\w']+$", "", (word or "").lower())


def load_model_word_confidences(dataset, model):
    """
    Returns:
        {
            dataset_index: [(word, confidence), ...]
        }
    """
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)

    result = {}

    for sample in samples:
        idx = sample.get("sample_index")
        if idx is None:
            continue

        segments = sample.get("segments") or []

        result[idx] = [
            (seg.get("word", ""), seg.get("confidence"))
            for seg in segments
            if seg.get("word")
        ]

    return result


def find_best_span(segment_text, word_conf_list, cursor):
    """
    Align one fixed sentence-like segment to the best local span in a
    source ASR transcript.

    Args:
        segment_text:
            Fixed fused sentence-like segment.

        word_conf_list:
            List of (word, confidence) tuples for one source ASR model.

        cursor:
            Start position for local sequential search. The cursor is
            carried forward across sentence segments within the same
            transcript to reduce alignment drift.

    Returns:
        mean_confidence:
            Mean native ASR confidence over the matched local span.

        alignment_score:
            Whole-span SequenceMatcher similarity between the fused
            segment and the matched source span.

        new_cursor:
            End index of the matched source span. If no match is found,
            the original cursor is returned.
    """
    anchor_tokens = re.findall(r"[\w']+", segment_text.lower())

    if not anchor_tokens:
        return None, 0.0, cursor

    source_tokens = [
        normalise_word(word)
        for word, _ in word_conf_list
    ]

    if not source_tokens:
        return None, 0.0, cursor

    sent_len = len(anchor_tokens)

    min_len = max(1, sent_len - MAX_EXTRA)
    max_len = sent_len + MAX_EXTRA

    search_end = min(
        len(source_tokens),
        cursor + sent_len * 3 + MAX_EXTRA,
    )

    best_start = None
    best_end = None
    best_score = 0.0

    for start in range(cursor, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len

            if end > len(source_tokens):
                continue

            score = difflib.SequenceMatcher(
                None,
                anchor_tokens,
                source_tokens[start:end],
            ).ratio()

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

        # Stop only after searching sufficiently beyond the best match
        # found so far, rather than relative to the original cursor.
        if (
            best_start is not None
            and best_score >= 0.88
            and start > best_start + 5
        ):
            break

    if best_start is None or best_end is None:
        return None, 0.0, cursor

    confidences = [
        conf
        for _, conf in word_conf_list[best_start:best_end]
        if conf is not None
    ]

    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    return mean_confidence, best_score, best_end


def compute_raw_segment_scores(dataset, split, model_confs):
    segments_data = load_segments(dataset, split)
    transcripts = segments_data.get("transcripts", [])

    per_transcript_results = []

    for transcript in transcripts:
        idx = transcript["dataset_index"]

        # One cursor per source model, carried across ALL segments in order.
        cursors_by_model = {
            model: 0
            for model in model_confs
        }

        seg_scores = []

        for seg in transcript.get("segments", []):
            per_model_confidence = {}
            per_model_alignment_score = {}

            # Always align and advance cursors, even if this segment has no
            # severity label. Otherwise later segments can drift because the
            # source-model cursor remains behind skipped content.
            for model, conf_data in model_confs.items():
                word_conf_list = conf_data.get(idx, [])
                cursor = cursors_by_model[model]

                score, alignment_score, new_cursor = find_best_span(
                    seg["segment"],
                    word_conf_list,
                    cursor,
                )

                cursors_by_model[model] = new_cursor
                per_model_alignment_score[model] = alignment_score

                if score is not None:
                    per_model_confidence[model] = score

            # Only labelled segments are saved/evaluated.
            if seg.get("severity") is None:
                continue

            seg_scores.append(
                {
                    "segment": seg["segment"],
                    "severity": seg.get("severity"),
                    "flagged": seg.get("flagged"),
                    "per_model_confidence": per_model_confidence,
                    "per_model_alignment_score": per_model_alignment_score,
                }
            )

        per_transcript_results.append(
            {
                "dataset_index": idx,
                "segments": seg_scores,
            }
        )

    return per_transcript_results


def fit_and_save_pooled_dev_stats():
    """
    Fit one mean/stdev pair per ASR model using pooled development data
    from the three in-domain datasets.

    Shetland is never used to fit normalization statistics.
    """
    print(
        "Fitting pooled dev normalization stats across "
        f"{IN_DOMAIN_DEV_DATASETS}..."
    )

    values_by_model = {
        model: []
        for model in ASR_MODELS
    }

    for dataset in IN_DOMAIN_DEV_DATASETS:
        print(f"\n  Loading {dataset} dev...")

        model_confs = {}

        for model in ASR_MODELS:
            try:
                model_confs[model] = load_model_word_confidences(
                    dataset,
                    model,
                )
            except FileNotFoundError:
                print(
                    f"    WARNING: no canonical file for "
                    f"{model}/{dataset} - skipping"
                )

        dev_results = compute_raw_segment_scores(
            dataset,
            "dev",
            model_confs,
        )

        for transcript in dev_results:
            for seg in transcript["segments"]:
                for model, score in seg["per_model_confidence"].items():
                    values_by_model[model].append(score)

        n_segments = sum(
            len(t["segments"])
            for t in dev_results
        )
        print(f"    contributed {n_segments} segments")

    stats = {}

    for model, values in values_by_model.items():
        if len(values) < 2:
            print(
                f"  WARNING: not enough pooled values for {model} "
                "to fit normalization stats"
            )
            continue

        stats[model] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values),
            "n": len(values),
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        POOLED_STATS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats, f, indent=2)

    print(
        f"\nSaved pooled normalization stats: "
        f"{POOLED_STATS_PATH}"
    )

    for model, values in stats.items():
        print(
            f"  {model}: "
            f"mean={values['mean']:.3f} "
            f"stdev={values['stdev']:.3f} "
            f"n={values['n']}"
        )

    return stats


def load_pooled_dev_stats():
    if not os.path.exists(POOLED_STATS_PATH):
        raise FileNotFoundError(
            f"No pooled normalization stats found at "
            f"{POOLED_STATS_PATH}. Run "
            "'python model_internal_confidence.py "
            "--fit-pooled-stats' first."
        )

    with open(
        POOLED_STATS_PATH,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def z_normalize(value, model_stats):
    if (
        model_stats is None
        or model_stats.get("stdev", 0) == 0
    ):
        return None

    return (
        value - model_stats["mean"]
    ) / model_stats["stdev"]


def run_dataset(dataset, split="test"):
    print(
        f"\n-- {dataset} | "
        f"model_internal_confidence split={split} --"
    )
    print(
        f"  Source models with native confidence: "
        f"{ASR_MODELS}"
    )

    model_confs = {}

    for model in ASR_MODELS:
        try:
            model_confs[model] = load_model_word_confidences(
                dataset,
                model,
            )
            print(
                f"  loaded word-confidence data for {model}"
            )
        except FileNotFoundError:
            print(
                f"  WARNING: no canonical file found for "
                f"{model}/{dataset} - skipping"
            )

    norm_stats = load_pooled_dev_stats()

    print(
        "  Using frozen pooled in-domain dev "
        "normalization stats"
    )

    raw_results = compute_raw_segment_scores(
        dataset,
        split,
        model_confs,
    )

    results = []

    for transcript in raw_results:
        out_segments = []

        for seg in transcript["segments"]:
            per_model_confidence = (
                seg["per_model_confidence"]
            )
            per_model_alignment_score = (
                seg["per_model_alignment_score"]
            )

            per_model_normalized = {}

            for model, raw_score in (
                per_model_confidence.items()
            ):
                z = z_normalize(
                    raw_score,
                    norm_stats.get(model),
                )

                if z is not None:
                    per_model_normalized[model] = z

            weighted_scores = []
            weights = []

            for model, z_score in (
                per_model_normalized.items()
            ):
                alignment_score = (
                    per_model_alignment_score.get(model, 0.0)
                )

                if alignment_score > 0:
                    weighted_scores.append(
                        z_score * alignment_score
                    )
                    weights.append(alignment_score)

            model_internal_confidence = (
                sum(weighted_scores) / sum(weights)
                if weights
                else None
            )

            out_segments.append(
                {
                    "segment": seg["segment"],
                    "severity": seg["severity"],
                    "flagged": seg["flagged"],
                    "per_model_confidence_raw": (
                        per_model_confidence
                    ),
                    "per_model_confidence_normalized": (
                        per_model_normalized
                    ),
                    "per_model_alignment_score": (
                        per_model_alignment_score
                    ),
                    "model_internal_confidence": (
                        model_internal_confidence
                    ),
                }
            )

        results.append(
            {
                "dataset_index": transcript["dataset_index"],
                "segments": out_segments,
            }
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"model_internal_{dataset}_{split}.json",
    )

    n_segments = sum(
        len(t["segments"])
        for t in results
    )

    n_scored = sum(
        1
        for t in results
        for seg in t["segments"]
        if seg.get("model_internal_confidence") is not None
    )

    output = {
        "dataset": dataset,
        "split": split,
        "method": "model_internal_asr_confidence",
        "alignment": (
            "local whole-span SequenceMatcher alignment "
            "with forward cursor"
        ),
        "normalization": (
            "per-model z-score fitted once on pooled "
            "CommonVoice + EdAcc + English Dialects dev data "
            "and frozen for all evaluation sets"
        ),
        "aggregation": (
            "alignment-score-weighted mean of normalized "
            "per-model sentence confidence"
        ),
        "asr_models_used": list(model_confs.keys()),
        "excluded_models": {
            "whisper_ft_chunked": (
                "no word-level confidence data available"
            )
        },
        "num_transcripts": len(results),
        "num_segments": n_segments,
        "num_segments_scored": n_scored,
        "transcripts": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"  {n_scored}/{n_segments} segments scored. "
        f"Saved: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="commonvoice",
        choices=DATASETS,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["dev", "test", "full"],
    )
    parser.add_argument(
        "--fit-pooled-stats",
        action="store_true",
        help=(
            "Fit and save pooled per-model normalization "
            "statistics from the three in-domain dev sets."
        ),
    )

    args = parser.parse_args()

    if args.fit_pooled_stats:
        fit_and_save_pooled_dev_stats()
    else:
        run_dataset(
            args.dataset,
            split=args.split,
        )


if __name__ == "__main__":
    main()
