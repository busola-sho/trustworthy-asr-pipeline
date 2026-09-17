"""
check_confidence_scales.py

Diagnostic: are Qwen, WhisperX, Parakeet, and Wav2Vec2's raw word-level
confidence values actually on comparable numeric scales? If one model's
values cluster systematically higher/lower than another's, averaging
raw values across models (even coverage-weighted) mixes different
calibration scales and the aggregate would be misleading - per-model
normalization would be needed first.

Run this BEFORE trusting model_internal_confidence.py's output.

Usage:
    python check_confidence_scales.py --dataset commonvoice
"""

import argparse
import statistics

from src.selector import find_canonical_file, load_samples

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


def get_confidence_values(dataset, model):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    values = []
    for s in samples:
        for seg in (s.get("segments") or []):
            conf = seg.get("confidence")
            if conf is not None:
                values.append(conf)
    return values


def summarize(values):
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "n": n,
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if n > 1 else 0.0,
        "min": sorted_vals[0],
        "p25": sorted_vals[int(n * 0.25)],
        "median": sorted_vals[int(n * 0.5)],
        "p75": sorted_vals[int(n * 0.75)],
        "max": sorted_vals[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    args = parser.parse_args()

    print(f"\nConfidence scale comparison - {args.dataset}\n")
    print(f"{'Model':<12}{'N':>8}{'Mean':>8}{'Stdev':>8}{'Min':>8}{'P25':>8}{'Median':>8}{'P75':>8}{'Max':>8}")
    print("-" * 76)

    summaries = {}
    for model in ASR_MODELS:
        try:
            values = get_confidence_values(args.dataset, model)
        except FileNotFoundError:
            print(f"{model:<12}  NO BENCHMARK FILE FOUND")
            continue
        s = summarize(values)
        if s is None:
            print(f"{model:<12}  NO CONFIDENCE VALUES FOUND")
            continue
        summaries[model] = s
        print(f"{model:<12}{s['n']:>8}{s['mean']:>8.3f}{s['stdev']:>8.3f}{s['min']:>8.3f}"
              f"{s['p25']:>8.3f}{s['median']:>8.3f}{s['p75']:>8.3f}{s['max']:>8.3f}")

    if len(summaries) < 2:
        print("\nNot enough models with data to compare scales.")
        return

    means = [s["mean"] for s in summaries.values()]
    spread = max(means) - min(means)
    print(f"\nSpread in means across models: {spread:.3f}")
    if spread > 0.15:
        print("WARNING: means differ substantially (>0.15) - these models' confidence")
        print("  scores may NOT be on comparable scales. Consider per-model normalization")
        print("  (e.g. z-score or min-max scaling per model) before aggregating.")
    else:
        print("Means are reasonably close - raw-value aggregation is likely defensible")
        print("  without additional normalization, but check the full distributions")
        print("  (stdev, percentiles) above too, not just the means.")


if __name__ == "__main__":
    main()
