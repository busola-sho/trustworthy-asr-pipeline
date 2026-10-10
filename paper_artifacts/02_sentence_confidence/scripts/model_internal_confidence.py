"""
rerunning/sentence_confidence/model_internal_confidence.py

Method 2: Model-internal ASR confidence. Aggregates native word-level
confidence, from all SOURCE MODELS WITH AVAILABLE CONFIDENCE DATA
(currently: Qwen, WhisperX, Parakeet, Wav2Vec2 - four, not five),
onto each fixed sentence segment. Motivated by established ASR
confidence-estimation principles, adapted to this sentence-level
ensemble setting (not a direct replication of a single paper).

LIMITATION: the fine-tuned Whisper model (whisper_ft_chunked) is
EXCLUDED - no word-level confidence data available (data-availability
gap, not a methodological choice).

PER-MODEL NORMALIZATION (POOLED, revised after two rounds of review):
raw confidence scales are NOT comparable across models - empirically
confirmed spreads of 0.16-0.29 in mean confidence across models on all
3 datasets (e.g. Qwen mean~0.95-0.99, heavily saturated; WhisperX
mean~0.67-0.83, much lower). Fixed via Z-SCORE NORMALIZATION per
model - but the normalization statistics (mu_m, sigma_m) are fitted
ONCE, POOLED across ALL THREE in-domain dev sets (CommonVoice + EdAcc
+ English Dialects) combined, per model - NOT per-dataset. This gives
Method 2 a single, consistent confidence scale across the whole
experiment: a Qwen z-score of 0 means "typical for Qwen" the same way
regardless of which dataset it's computed on. The SAME frozen pooled
stats are then applied unchanged to every test set AND to Shetland -
Shetland never fits its own stats, since it is the out-of-domain
confirmation set and fitting normalization on it would undermine that
design (there is also no Shetland dev split to fit from in the first
place).

Pipeline: native word confidence -> segment-level mean per model ->
z-score normalize using POOLED in-domain dev mu/sigma (frozen) ->
coverage-weighted mean across models.

Usage:
    # one-time step, run before anything else:
    python model_internal_confidence.py --fit-pooled-stats

    # then, for every dataset/split (including shetland):
    python model_internal_confidence.py --dataset commonvoice --split dev
    python model_internal_confidence.py --dataset commonvoice --split test
    python model_internal_confidence.py --dataset shetland --split full
"""

import json
import os
import re
import argparse
import statistics

from src.selector import find_canonical_file, load_samples

SEGMENTS_DIR = "writeup_results/sentence_confidence/segments"
OUTPUT_DIR = "writeup_results/sentence_confidence/model_internal_confidence"
POOLED_STATS_PATH = "writeup_results/sentence_confidence/model_internal_confidence/norm_stats_pooled.json"
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]
IN_DOMAIN_DEV_DATASETS = ["commonvoice", "edacc", "english_dialects"]  # shetland never contributes to fitting

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]


def load_segments(dataset, split):
    path = os.path.join(SEGMENTS_DIR, f"sentence_segments_{dataset}_{split}.json")
    return json.load(open(path))


def load_model_word_confidences(dataset, model):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    result = {}
    for s in samples:
        idx = s.get("sample_index")
        if idx is None:
            continue
        segs = s.get("segments") or []
        result[idx] = [(seg.get("word", ""), seg.get("confidence"))
                        for seg in segs if seg.get("word")]
    return result


def match_segment_confidence(segment_text, word_conf_list, used_positions):
    segment_words = re.findall(r"[\w']+", segment_text.lower())
    if not segment_words:
        return None, 0.0

    matched_confidences = []
    n_matched = 0
    search_from = 0
    for word in segment_words:
        found_at = None
        for i in range(search_from, len(word_conf_list)):
            if i in used_positions:
                continue
            candidate_word, conf = word_conf_list[i]
            if candidate_word.lower().strip(".,!?;:\"'") == word:
                found_at = i
                break
        if found_at is not None:
            _, conf = word_conf_list[found_at]
            if conf is not None:
                matched_confidences.append(conf)
            used_positions.add(found_at)
            search_from = found_at + 1
            n_matched += 1

    coverage = n_matched / len(segment_words)
    mean_confidence = (sum(matched_confidences) / len(matched_confidences)
                       if matched_confidences else None)
    return mean_confidence, coverage


def compute_raw_segment_scores(dataset, split, model_confs):
    segments_data = load_segments(dataset, split)
    transcripts = segments_data.get("transcripts", [])

    per_transcript_results = []
    for t in transcripts:
        idx = t["dataset_index"]
        used_positions_by_model = {m: set() for m in model_confs}

        seg_scores = []
        for seg in t.get("segments", []):
            if seg.get("severity") is None:
                continue

            per_model_confidence = {}
            per_model_coverage = {}
            for model, conf_data in model_confs.items():
                word_conf_list = conf_data.get(idx, [])
                score, coverage = match_segment_confidence(seg["segment"], word_conf_list,
                                                            used_positions_by_model[model])
                per_model_coverage[model] = coverage
                if score is not None:
                    per_model_confidence[model] = score

            seg_scores.append({
                "segment": seg["segment"],
                "severity": seg.get("severity"),
                "flagged": seg.get("flagged"),
                "per_model_confidence": per_model_confidence,
                "per_model_coverage": per_model_coverage,
            })

        per_transcript_results.append({"dataset_index": idx, "segments": seg_scores})

    return per_transcript_results


def fit_and_save_pooled_dev_stats():
    """Fits z-score normalization stats (mean, stdev) per model, POOLED
    across ALL THREE in-domain dev sets combined - one set of stats for
    the whole experiment, not one per dataset. Shetland never
    contributes (out-of-domain confirmation set, no dev split of its
    own). Saved once to POOLED_STATS_PATH; every subsequent run of any
    dataset/split loads these, never refits."""
    print("Fitting POOLED dev normalization stats across "
          f"{IN_DOMAIN_DEV_DATASETS}...")

    values_by_model = {m: [] for m in ASR_MODELS}

    for dataset in IN_DOMAIN_DEV_DATASETS:
        print(f"\n  Loading {dataset} dev...")
        model_confs = {}
        for model in ASR_MODELS:
            try:
                model_confs[model] = load_model_word_confidences(dataset, model)
            except FileNotFoundError:
                print(f"    WARNING: no benchmark file for {model}/{dataset} - skipping for this dataset")

        dev_results = compute_raw_segment_scores(dataset, "dev", model_confs)
        for t in dev_results:
            for seg in t["segments"]:
                for model, score in seg["per_model_confidence"].items():
                    values_by_model[model].append(score)
        print(f"    contributed {sum(len(t['segments']) for t in dev_results)} segments")

    stats = {}
    for model, values in values_by_model.items():
        if len(values) < 2:
            print(f"  WARNING: not enough pooled values for {model} to fit normalization stats")
            continue
        stats[model] = {"mean": statistics.mean(values), "stdev": statistics.stdev(values), "n": len(values)}

    os.makedirs(os.path.dirname(POOLED_STATS_PATH), exist_ok=True)
    with open(POOLED_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved pooled normalization stats: {POOLED_STATS_PATH}")
    for model, s in stats.items():
        print(f"  {model}: mean={s['mean']:.3f}  stdev={s['stdev']:.3f}  n={s['n']} (pooled across all 3 in-domain dev sets)")

    return stats


def load_pooled_dev_stats():
    if not os.path.exists(POOLED_STATS_PATH):
        raise FileNotFoundError(
            f"No pooled normalization stats found at {POOLED_STATS_PATH}. "
            f"Run 'python model_internal_confidence.py --fit-pooled-stats' first."
        )
    return json.load(open(POOLED_STATS_PATH))


def z_normalize(value, model_stats):
    if model_stats is None or model_stats.get("stdev", 0) == 0:
        return None
    return (value - model_stats["mean"]) / model_stats["stdev"]


def run_dataset(dataset, split="test"):
    print(f"\n-- {dataset} | model_internal_confidence split={split} --")
    print(f"  Source models (confidence available): {ASR_MODELS}")

    model_confs = {}
    for model in ASR_MODELS:
        try:
            model_confs[model] = load_model_word_confidences(dataset, model)
            print(f"  loaded word-confidence data for {model}")
        except FileNotFoundError:
            print(f"  WARNING: no benchmark file found for {model}/{dataset} - skipping this model")

    # ALWAYS load the pooled stats - never fit per-dataset, never refit
    # for dev either (dev uses the SAME pooled stats as everything else,
    # so numbers stay comparable across the whole experiment).
    norm_stats = load_pooled_dev_stats()
    print(f"  Using pooled in-domain dev normalization stats (frozen)")

    raw_results = compute_raw_segment_scores(dataset, split, model_confs)

    results = []
    for t in raw_results:
        out_segments = []
        for seg in t["segments"]:
            per_model_confidence = seg["per_model_confidence"]
            per_model_coverage = seg["per_model_coverage"]

            per_model_normalized = {}
            for model, raw_score in per_model_confidence.items():
                z = z_normalize(raw_score, norm_stats.get(model))
                if z is not None:
                    per_model_normalized[model] = z

            weighted_scores = []
            weights = []
            for model, z_score in per_model_normalized.items():
                coverage = per_model_coverage[model]
                if coverage > 0:
                    weighted_scores.append(z_score * coverage)
                    weights.append(coverage)

            mean_confidence_normalized = (sum(weighted_scores) / sum(weights)
                                          if weights else None)

            out_segments.append({
                "segment": seg["segment"],
                "severity": seg["severity"],
                "flagged": seg["flagged"],
                "per_model_confidence_raw": per_model_confidence,
                "per_model_confidence_normalized": per_model_normalized,
                "per_model_coverage": per_model_coverage,
                "model_internal_confidence": mean_confidence_normalized,
            })

        results.append({"dataset_index": t["dataset_index"], "segments": out_segments})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"model_internal_{dataset}_{split}.json")

    n_segments = sum(len(t["segments"]) for t in results)
    n_scored = sum(1 for t in results for s in t["segments"]
                   if s.get("model_internal_confidence") is not None)

    output = {
        "dataset": dataset,
        "split": split,
        "method": "model_internal_asr_confidence",
        "normalization": "per-model z-score, POOLED across all 3 in-domain dev sets, frozen and applied unchanged everywhere (including Shetland)",
        "aggregation": "coverage-weighted mean of normalized per-model scores",
        "asr_models_used": list(model_confs.keys()),
        "excluded_models": {"whisper_ft_chunked": "no word-level confidence data available"},
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
    parser.add_argument("--fit-pooled-stats", action="store_true",
                        help="One-time step: fit and save pooled normalization stats "
                             "across all 3 in-domain dev sets. Run this before anything else.")
    args = parser.parse_args()

    if args.fit_pooled_stats:
        fit_and_save_pooled_dev_stats()
    else:
        run_dataset(args.dataset, split=args.split)


if __name__ == "__main__":
    main()
