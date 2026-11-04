"""Four-source sentence confidence: model_internal_confidence."""

from asr_pipeline.confidence.paths import SentenceConfidenceFiles

import argparse
import difflib
import json
import os
import re
import statistics

from asr_pipeline.confidence.inputs import (
    SOURCE_MODELS,
    load_word_confidence_samples,
    load_confidence_segments,
)


DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
IN_DOMAIN_DEV_DATASETS = ["commonvoice", "edacc", "english_dialects"]

ASR_MODELS = list(SOURCE_MODELS)

MAX_EXTRA = 10


def load_segments(dataset, split, paths=None):
    paths = paths or SentenceConfidenceFiles()
    return load_confidence_segments(dataset, split, paths=paths)


def normalise_word(word):
    return re.sub(r"^[^\w']+|[^\w']+$", "", (word or "").lower())


def load_model_word_confidences(dataset, model, split, *, fusion_file=None, paths=None):
    """
    Returns:
        {
            dataset_index: [(word, confidence), ...]
        }
    """
    paths = paths or SentenceConfidenceFiles()
    samples = load_word_confidence_samples(
        dataset, model, split, fusion_file=fusion_file, paths=paths
    )

    result = {}

    for sample in samples:
        transcript_index = sample.get("sample_index")
        if transcript_index is None:
            continue

        segments = sample.get("segments") or []

        result[transcript_index] = [
            (segment.get("word", ""), segment.get("confidence"))
            for segment in segments
            if segment.get("word")
        ]

    return result


def find_best_span(segment_text, word_confidences, cursor):
    """Find the source words that best match a sentence and average their confidence.

    Return the confidence, match score and next cursor. Search forward through the
    source words. Move the cursor even when a sentence has no evaluation label."""
    anchor_tokens = re.findall(r"[\w']+", segment_text.lower())

    if not anchor_tokens:
        return None, 0.0, cursor

    source_tokens = [normalise_word(word) for word, _ in word_confidences]

    if not source_tokens:
        return None, 0.0, cursor

    sentence_word_count = len(anchor_tokens)

    min_len = max(1, sentence_word_count - MAX_EXTRA)
    max_len = sentence_word_count + MAX_EXTRA

    search_end = min(
        len(source_tokens),
        cursor + sentence_word_count * 3 + MAX_EXTRA,
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
        if best_start is not None and best_score >= 0.88 and start > best_start + 5:
            break

    if best_start is None or best_end is None:
        return None, 0.0, cursor

    confidences = [
        conf for _, conf in word_confidences[best_start:best_end] if conf is not None
    ]

    mean_confidence = sum(confidences) / len(confidences) if confidences else None

    return mean_confidence, best_score, best_end


def compute_raw_segment_scores(
    dataset,
    split,
    word_confidences_by_model,
    *,
    segments_data=None,
    include_unlabelled=False,
    paths=None,
):
    paths = paths or SentenceConfidenceFiles()
    if segments_data is None:
        segments_data = load_segments(dataset, split, paths=paths)
    transcripts = segments_data.get("transcripts", [])

    per_transcript_results = []

    for transcript in transcripts:
        transcript_index = transcript["dataset_index"]

        # One cursor per source model, carried across ALL segments in order.
        cursors_by_model = {model: 0 for model in word_confidences_by_model}

        seg_scores = []

        for segment in transcript.get("segments", []):
            per_model_confidence = {}
            per_model_alignment_score = {}

            # Always align and advance cursors, even if this segment has no
            # severity label. Otherwise later segments can drift because the
            # source-model cursor remains behind skipped content.
            for model, conf_data in word_confidences_by_model.items():
                word_confidences = conf_data.get(transcript_index, [])
                cursor = cursors_by_model[model]

                score, alignment_score, new_cursor = find_best_span(
                    segment["segment"],
                    word_confidences,
                    cursor,
                )

                cursors_by_model[model] = new_cursor
                per_model_alignment_score[model] = alignment_score

                if score is not None:
                    per_model_confidence[model] = score

            # Only labelled segments are saved/evaluated.
            if segment.get("severity") is None and not include_unlabelled:
                continue

            seg_scores.append(
                {
                    "segment": segment["segment"],
                    "severity": segment.get("severity"),
                    "flagged": segment.get("flagged"),
                    "per_model_confidence": per_model_confidence,
                    "per_model_alignment_score": per_model_alignment_score,
                }
            )

        per_transcript_results.append(
            {
                "dataset_index": transcript_index,
                "segments": seg_scores,
            }
        )

    return per_transcript_results


def fit_development_statistics(paths=None):
    """
    Fit one mean/stdev pair per ASR model using pooled development data
    from the three in-domain datasets.

    Shetland is never used to fit normalization statistics.
    """
    paths = paths or SentenceConfidenceFiles()
    print(f"Fitting pooled dev normalization stats across {IN_DOMAIN_DEV_DATASETS}...")

    values_by_model = {model: [] for model in ASR_MODELS}

    for dataset in IN_DOMAIN_DEV_DATASETS:
        print(f"\n  Loading {dataset} dev...")

        word_confidences_by_model = {}

        for model in ASR_MODELS:
            word_confidences_by_model[model] = load_model_word_confidences(
                dataset, model, "dev", paths=paths
            )

        dev_results = compute_raw_segment_scores(
            dataset, "dev", word_confidences_by_model, paths=paths
        )

        for transcript in dev_results:
            for segment in transcript["segments"]:
                for model, score in segment["per_model_confidence"].items():
                    values_by_model[model].append(score)

        segment_count = sum(len(t["segments"]) for t in dev_results)
        print(f"    contributed {segment_count} segments")

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

    os.makedirs(paths.internal_dir, exist_ok=True)

    with open(
        paths.development_statistics_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved pooled normalization stats: {paths.development_statistics_file}")

    for model, values in stats.items():
        print(
            f"  {model}: "
            f"mean={values['mean']:.3f} "
            f"stdev={values['stdev']:.3f} "
            f"n={values['n']}"
        )

    return stats


def load_development_statistics(paths=None):
    paths = paths or SentenceConfidenceFiles()
    if not os.path.exists(paths.development_statistics_file):
        raise FileNotFoundError(
            f"No pooled normalization stats found at "
            f"{paths.development_statistics_file}. Run "
            "'python model_internal_confidence.py "
            "--fit-pooled-stats' first."
        )

    with open(
        paths.development_statistics_file,
        encoding="utf-8",
    ) as f:
        return json.load(f)


def z_normalize(value, model_stats):
    if model_stats is None or model_stats.get("stdev", 0) == 0:
        return None

    return (value - model_stats["mean"]) / model_stats["stdev"]


def run_dataset(dataset, split="test", paths=None):
    paths = paths or SentenceConfidenceFiles()
    print(f"\n-- {dataset} | model_internal_confidence split={split} --")
    print(f"  Source models with native confidence: {ASR_MODELS}")

    word_confidences_by_model = {}

    for model in ASR_MODELS:
        word_confidences_by_model[model] = load_model_word_confidences(
            dataset, model, split, paths=paths
        )
        print(f"  loaded word-confidence data for {model}")

    norm_stats = load_development_statistics(paths=paths)

    print("  Using frozen pooled in-domain dev normalization stats")

    raw_results = compute_raw_segment_scores(
        dataset, split, word_confidences_by_model, paths=paths
    )

    results = []

    for transcript in raw_results:
        out_segments = []

        for segment in transcript["segments"]:
            per_model_confidence = segment["per_model_confidence"]
            per_model_alignment_score = segment["per_model_alignment_score"]

            per_model_normalized = {}

            for model, raw_score in per_model_confidence.items():
                z = z_normalize(
                    raw_score,
                    norm_stats.get(model),
                )

                if z is not None:
                    per_model_normalized[model] = z

            weighted_scores = []
            weights = []

            for model, z_score in per_model_normalized.items():
                alignment_score = per_model_alignment_score.get(model, 0.0)

                if alignment_score > 0:
                    weighted_scores.append(z_score * alignment_score)
                    weights.append(alignment_score)

            asr_confidence = sum(weighted_scores) / sum(weights) if weights else None

            out_segments.append(
                {
                    "segment": segment["segment"],
                    "severity": segment["severity"],
                    "flagged": segment["flagged"],
                    "per_model_confidence_raw": (per_model_confidence),
                    "per_model_confidence_normalized": (per_model_normalized),
                    "per_model_alignment_score": (per_model_alignment_score),
                    "model_internal_confidence": (asr_confidence),
                }
            )

        results.append(
            {
                "dataset_index": transcript["dataset_index"],
                "segments": out_segments,
            }
        )

    os.makedirs(paths.internal_dir, exist_ok=True)

    output_path = os.path.join(
        paths.internal_dir,
        f"model_internal_{dataset}_{split}.json",
    )

    segment_count = sum(len(t["segments"]) for t in results)

    scored_count = sum(
        1
        for t in results
        for segment in t["segments"]
        if segment.get("model_internal_confidence") is not None
    )

    output = {
        "dataset": dataset,
        "split": split,
        "method": "model_internal_asr_confidence",
        "alignment": ("local whole-span SequenceMatcher alignment with forward cursor"),
        "normalization": (
            "per-model z-score fitted once on pooled "
            "CommonVoice + EdAcc + English Dialects dev data "
            "and frozen for all evaluation sets"
        ),
        "aggregation": (
            "alignment-score-weighted mean of normalized per-model sentence confidence"
        ),
        "asr_models_used": list(word_confidences_by_model.keys()),
        "excluded_models": {},
        "num_transcripts": len(results),
        "num_segments": segment_count,
        "num_segments_scored": scored_count,
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

    print(f"  {scored_count}/{segment_count} segments scored. Saved: {output_path}")


def main(argv=None, paths=None):
    paths = paths or SentenceConfidenceFiles()
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

    args = parser.parse_args(argv)

    if args.fit_pooled_stats:
        fit_development_statistics(paths=paths)
    else:
        run_dataset(args.dataset, split=args.split, paths=paths)


if __name__ == "__main__":
    main()
