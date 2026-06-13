"""
summarise_all_results.py

Reads all combination benchmark JSONs and prints a comparison table
showing WER and MAR vs single-model baselines.

Usage:
    python scripts/summarise_all_results.py
    python scripts/summarise_all_results.py --dataset commonvoice
"""

import json
import os
import glob
import argparse

COMBINATION_DIR = "results/combinations"

BASELINES = {
    "commonvoice":      {"wer": 0.1884, "mar": 0.7368},
    "edacc":            {"wer": 0.1692, "mar": 0.2475},
    "english_dialects": {"wer": 0.0446, "mar": 0.1085},
}

def load_file(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")
        return None

    fname    = os.path.basename(path)
    approach = "context" if fname.startswith("context_") else "naive"
    dataset  = data.get("dataset", "—")
    selector = data.get("selector", "—")
    prompt   = data.get("selector_prompt", "—") if approach == "naive" else "context"
    wer_val  = data.get("corpus_wer")
    mar_val  = data.get("meaning_alteration_rate")
    n        = data.get("num_samples", 0)
    in_prog  = "progress" in data and wer_val is None

    return {
        "fname":    fname,
        "approach": approach,
        "dataset":  dataset,
        "selector": selector,
        "prompt":   prompt,
        "wer":      wer_val,
        "mar":      mar_val,
        "n":        n,
        "in_prog":  in_prog,
    }

def print_table(rows, dataset, baseline):
    b_wer = baseline["wer"]
    b_mar = baseline["mar"]

    print(f"\n{'='*85}")
    print(f"Dataset: {dataset}  |  Baseline (Qwen3-ASR): WER={b_wer*100:.2f}%  MAR={b_mar*100:.2f}%")
    print(f"{'='*85}")
    print(f"  {'Approach':<10} {'Selector':<10} {'Prompt':<8} {'N':>5} {'WER':>8} {'ΔWER':>8} {'MAR':>8} {'ΔMAR':>8}  Notes")
    print(f"  {'-'*80}")

    # sort by MAR
    rows_sorted = sorted(rows, key=lambda x: x["mar"] if x["mar"] is not None else 99)

    for r in rows_sorted:
        wer_str  = f"{r['wer']*100:.2f}%" if r["wer"] is not None else "—"
        mar_str  = f"{r['mar']*100:.2f}%" if r["mar"] is not None else "—"
        d_wer    = f"{(r['wer']-b_wer)*100:+.2f}pp" if r["wer"] is not None else "—"
        d_mar    = f"{(r['mar']-b_mar)*100:+.2f}pp" if r["mar"] is not None else "—"

        notes = []
        if r["in_prog"]:
            notes.append("in-progress")
        if r["wer"] is not None and r["wer"] < b_wer:
            notes.append("★WER")
        if r["mar"] is not None and r["mar"] < b_mar:
            notes.append("★MAR")
        notes_str = " ".join(notes)

        print(f"  {r['approach']:<10} {r['selector']:<10} {r['prompt']:<8} "
              f"{r['n']:>5} {wer_str:>8} {d_wer:>8} {mar_str:>8} {d_mar:>8}  {notes_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "commonvoice", "edacc", "english_dialects"])
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(COMBINATION_DIR, "*.json")))
    if not files:
        print(f"No files found in {COMBINATION_DIR}")
        return

    results = []
    for path in files:
        r = load_file(path)
        if r:
            results.append(r)

    datasets = (
        ["commonvoice", "edacc", "english_dialects"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        rows     = [r for r in results if r["dataset"] == dataset]
        baseline = BASELINES.get(dataset)
        if not rows or not baseline:
            continue
        print_table(rows, dataset, baseline)

    print()

if __name__ == "__main__":
    main()