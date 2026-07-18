"""
extract_samples_for_annotation.py

Pulls ref/hyp pairs (with existing binary MAR verdicts) from individual-model
benchmark JSON files, and builds a stratified candidate pool for human
severity annotation — oversampling likely-error sentences so severity 3-4
cases aren't too rare to evaluate judges on properly.

Expects files shaped like:
    {model}_{dataset}_{timestamp}.json
    {
      "model": "...",
      "dataset": "...",
      "samples": [
        {"ref": "...", "hyp": "...", "sample_WER": 0.1, "<judge>_verdict_<pass>": true/false, ...}
      ]
    }

Usage:
    python extract_samples_for_annotation.py \
        --input-dir results/benchmarks/main \
        --output candidate_pool.json \
        --target-total 150 \
        --error-fraction 0.6
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

VERDICT_KEY_SUBSTR = "verdict"


def find_verdict_keys(sample: dict) -> list:
    return [k for k in sample.keys() if VERDICT_KEY_SUBSTR in k.lower()]


def is_flagged_error(sample: dict, verdict_keys: list) -> bool:
    """Union rule: flagged if ANY judge verdict is True.
    Deliberately high-recall — you'll review every flagged case during
    annotation anyway, so over-sampling candidates costs less than
    under-sampling real errors."""
    return any(bool(sample.get(k)) for k in verdict_keys)


def load_benchmark_file(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)

    model = data.get("model", path.stem)
    dataset = data.get("dataset", "unknown")
    records = []

    for idx, sample in enumerate(data.get("samples", [])):
        verdict_keys = find_verdict_keys(sample)
        record = {
            "uid": f"{path.stem}__{idx}",
            "model": model,
            "dataset": dataset,
            "source_file": path.name,
            "sample_index": idx,
            "ref": sample.get("ref", ""),
            "hyp": sample.get("hyp", ""),
            "sample_wer": sample.get("sample_WER"),
            "existing_verdicts": {k: sample.get(k) for k in verdict_keys},
            "flagged_likely_error": is_flagged_error(sample, verdict_keys),
        }
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Directory containing benchmark JSON files")
    parser.add_argument("--models", nargs="+",
                         default=["whisper", "qwen", "parakeet", "wav2vec2"],
                         help="Model filename prefixes to include")
    parser.add_argument("--datasets", nargs="+",
                         default=["commonvoice", "edacc", "english_dialects"],
                         help="Dataset filename substrings to include. "
                              "Shetland excluded by default (held-out test set).")
    parser.add_argument("--output", type=Path, default=Path("candidate_pool.json"))
    parser.add_argument("--target-total", type=int, default=150)
    parser.add_argument("--error-fraction", type=float, default=0.6,
                         help="Fraction of final sample drawn from the flagged-error pool")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_files = sorted(args.input_dir.glob("*.json"))
    selected_files = [
        f for f in all_files
        if any(f.stem.startswith(m) for m in args.models)
        and any(d in f.stem for d in args.datasets)
    ]

    if not selected_files:
        raise SystemExit(f"No matching files found in {args.input_dir}")

    print(f"Found {len(selected_files)} matching files:")
    for f in selected_files:
        print(f"  {f.name}")

    all_records = []
    for f in selected_files:
        all_records.extend(load_benchmark_file(f))

    # --- Dedupe by (dataset, ref) so the same underlying ground-truth sentence
    # never appears twice just because different models transcribed it.
    # Stratum is decided at the ref level: a ref counts as "error" if ANY
    # model's hyp for it got flagged, even if other models got it right.
    groups = defaultdict(list)
    for r in all_records:
        groups[(r["dataset"], r["ref"])].append(r)

    print(f"\nTotal (model, ref) records: {len(all_records)}")
    print(f"Unique (dataset, ref) groups: {len(groups)}")

    error_groups = []
    correct_groups = []
    for key, recs in groups.items():
        flagged_recs = [r for r in recs if r["flagged_likely_error"]]
        if flagged_recs:
            error_groups.append(flagged_recs)   # candidates to pick from = the flagged ones
        else:
            correct_groups.append(recs)         # none flagged, pick from any

    print(f"  Unique refs with >=1 flagged model: {len(error_groups)}")
    print(f"  Unique refs with 0 flagged models:  {len(correct_groups)}")

    n_error_target = int(args.target_total * args.error_fraction)
    n_correct_target = args.target_total - n_error_target

    if len(error_groups) < n_error_target:
        print(f"\nWARNING: only {len(error_groups)} error-flagged unique refs available, "
              f"wanted {n_error_target}. Taking all of them.")
        n_error_target = len(error_groups)
        n_correct_target = args.target_total - n_error_target

    if len(correct_groups) < n_correct_target:
        print(f"\nWARNING: only {len(correct_groups)} correct-only unique refs available, "
              f"wanted {n_correct_target}. Taking all of them.")
        n_correct_target = len(correct_groups)

    sampled_error_groups = random.sample(error_groups, n_error_target)
    sampled_correct_groups = random.sample(correct_groups, n_correct_target)

    # One representative record per group (one model's hyp per unique ref)
    final_pool = [random.choice(g) for g in sampled_error_groups]
    final_pool += [random.choice(g) for g in sampled_correct_groups]
    random.shuffle(final_pool)  # so annotation order doesn't reveal the stratum

    for r in final_pool:
        r["human_severity"] = None
        r["human_note"] = ""

    with open(args.output, "w") as f:
        json.dump(final_pool, f, indent=2)

    print(f"\nWrote {len(final_pool)} candidate sentences to {args.output}")
    print(f"  ({len(sampled_error_groups)} likely-error, {len(sampled_correct_groups)} likely-correct)")

    breakdown = defaultdict(int)
    for r in final_pool:
        breakdown[(r["dataset"], r["model"])] += 1
    print("\nBreakdown by (dataset, model):")
    for k, v in sorted(breakdown.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()