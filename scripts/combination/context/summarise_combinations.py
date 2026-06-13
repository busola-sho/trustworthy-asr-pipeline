"""
summarise_combinations.py

Reads all combination benchmark JSONs and outputs a summary table
showing WER and MAR for each selector/prompt/judge combination.
Handles in-progress files gracefully.

Usage:
    python scripts/summarise_combinations.py
    python scripts/summarise_combinations.py --dataset commonvoice
"""

import json
import os
import glob
import argparse
from jiwer import wer

COMBINATION_DIR  = "results/combinations"
BENCHMARKS_DIR   = "benchmarks"

# Best individual model baselines (Qwen3-ASR, Qwen P2 judge)
BASELINES = {
    "commonvoice":      {"wer": 0.1884, "mar": 0.7368},
    "edacc":            {"wer": 0.1692, "mar": 0.2475},
    "english_dialects": {"wer": 0.0446, "mar": 0.1085},
}

def normalise(text: str) -> str:
    import re
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = re.sub(r"[^\w\s']", '', text)
    return text.strip()

def compute_from_samples(samples):
    """Recompute corpus WER and MAR from samples — handles in-progress files."""
    valid = [
        s for s in samples
        if not s.get("skipped") and not s.get("error")
        and s.get("sample_WER") is not None
        and s.get("hyp")
    ]
    if not valid:
        return None, None, 0

    corpus_wer = wer(
        [normalise(s["ref"]) for s in valid],
        [normalise(s["hyp"]) for s in valid]
    )
    mar = sum(1 for s in valid if s.get("qwen_verdict_p2")) / len(valid)
    return corpus_wer, mar, len(valid)

def load_result(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None

    samples = data.get("samples", [])
    if not samples:
        return None

    # use stored metrics if complete, otherwise recompute
    corpus_wer = data.get("corpus_wer")
    mar        = data.get("meaning_alteration_rate")
    n          = data.get("num_samples", 0)

    # if in-progress (progress key exists, no corpus_wer), recompute
    if "progress" in data and corpus_wer is None:
        corpus_wer, mar, n = compute_from_samples(samples)
        in_progress = True
    else:
        in_progress = data.get("corpus_wer") is None
        if in_progress or n == 0:
            corpus_wer, mar, n = compute_from_samples(samples)

    return {
        "selector":        data.get("selector", "—"),
        "selector_prompt": data.get("selector_prompt", "—"),
        "judge":           data.get("judge", "—"),
        "dataset":         data.get("dataset", "—"),
        "wer":             corpus_wer,
        "mar":             mar,
        "n":               n,
        "in_progress":     "progress" in data and data.get("corpus_wer") is None,
        "post_processed":  data.get("post_processed", False),
        "fname":           os.path.basename(path),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        help="commonvoice, edacc, english_dialects, or all")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(COMBINATION_DIR, "*.json")))
    if not files:
        print(f"No files found in {COMBINATION_DIR}")
        return

    results = []
    for path in files:
        r = load_result(path)
        if r:
            results.append(r)

    datasets = (
        ["commonvoice", "english_dialects", "edacc"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        rows = [r for r in results if r["dataset"] == dataset]
        if not rows:
            continue

        baseline = BASELINES.get(dataset, {})
        baseline_wer = baseline.get("wer")
        baseline_mar = baseline.get("mar")

        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}  |  Baseline (Qwen3-ASR): WER={baseline_wer*100:.2f}%  MAR={baseline_mar*100:.2f}%")
        print(f"{'='*80}")
        print(f"{'Selector':<10} {'Prompt':<7} {'Judge':<8} {'N':>5} {'WER':>8} {'ΔWER':>7} {'MAR':>8} {'ΔMAR':>7} {'Notes'}")
        print("-" * 80)

        # sort by WER
        rows_sorted = sorted(rows, key=lambda x: x["wer"] if x["wer"] is not None else 99)

        for r in rows_sorted:
            wer_str  = f"{r['wer']*100:.2f}%"  if r["wer"] is not None else "—"
            mar_str  = f"{r['mar']*100:.2f}%"  if r["mar"] is not None else "—"

            delta_wer = ""
            delta_mar = ""
            if r["wer"] is not None and baseline_wer is not None:
                d = (r["wer"] - baseline_wer) * 100
                delta_wer = f"{d:+.2f}pp"
            if r["mar"] is not None and baseline_mar is not None:
                d = (r["mar"] - baseline_mar) * 100
                delta_mar = f"{d:+.2f}pp"

            notes = []
            if r["in_progress"]:
                notes.append("in-progress")
            if r["post_processed"]:
                notes.append("pp")
            notes_str = ", ".join(notes)

            # bold if beats baseline
            beat = r["wer"] is not None and baseline_wer is not None and r["wer"] < baseline_wer
            marker = " ★" if beat else ""

            print(f"{r['selector']:<10} {r['selector_prompt']:<7} {r['judge']:<8} "
                  f"{r['n']:>5} {wer_str:>8} {delta_wer:>7} {mar_str:>8} {delta_mar:>7} "
                  f"{notes_str}{marker}")

    print()

if __name__ == "__main__":
    main()