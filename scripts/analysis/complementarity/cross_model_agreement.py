"""
cross_model_agreement.py

For each dataset, analyses which combinations of models get each clip right.
Uses sample_WER threshold to define "correct" per clip per model.

Outputs:
    analysis/agreement_{dataset}.json   — full combination counts
    analysis/plots/agreement_*.png      — bar charts

Usage:
    python scripts/cross_model_agreement.py
    python scripts/cross_model_agreement.py --dataset commonvoice
    python scripts/cross_model_agreement.py --threshold 0.3
"""

import json
import os
import argparse
from collections import Counter
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────────

BENCHMARKS_DIR = "benchmarks"
OUTPUT_DIR     = "analysis"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "qwen_english_dialects_20260525_000627.json",
    ("whisper",  "commonvoice"):      "whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_20260525_110315.json",
    ("parakeet", "commonvoice"):      "parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_20260524_234807.json",
    ("wav2vec2", "commonvoice"):      "wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "wav2vec2_english_dialects_20260526_073439.json",
}

MODELS   = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

MODEL_LABELS = {
    "qwen":     "Qwen3-ASR",
    "whisper":  "Whisper",
    "parakeet": "Parakeet",
    "wav2vec2": "wav2vec2",
}

# ── Core analysis ──────────────────────────────────────────────────────────────

def analyse_dataset(dataset: str, threshold: float) -> dict:
    print(f"\nAnalysing {dataset} (threshold={threshold})...")

    # load all 4 model files
    model_samples = {}
    for model in MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            data = json.load(f)
        model_samples[model] = data["samples"]

    n_total = len(model_samples["qwen"])

    # ── Per-clip analysis ──────────────────────────────────────────────────────
    combination_counts = Counter()  # frozenset of correct models -> count
    n_correct_counts   = Counter()  # number of correct models (0-4) -> count
    model_win_counts   = Counter()  # model -> clips where it's the sole best
    model_correct      = Counter()  # model -> total clips it got right
    n_valid = 0

    for i in range(n_total):
        ref = model_samples["qwen"][i]["ref"]
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        # get WER for each model on this clip
        wers = {}
        for model in MODELS:
            wer = model_samples[model][i].get("sample_WER")
            if wer is not None:
                wers[model] = wer

        if not wers:
            continue

        n_valid += 1

        # which models got it "right" (below threshold)
        correct_models = frozenset(m for m, w in wers.items() if w <= threshold)

        combination_counts[correct_models] += 1
        n_correct_counts[len(correct_models)] += 1

        for m in correct_models:
            model_correct[m] += 1

        # which model has the best WER on this clip
        best_model = min(wers, key=wers.get)
        model_win_counts[best_model] += 1

    print(f"  {n_valid} valid clips")

    # ── Build combination table ────────────────────────────────────────────────
    # convert frozensets to readable labels, sort by count
    combo_table = []
    for combo, count in combination_counts.most_common():
        if len(combo) == 0:
            label = "None correct"
        elif len(combo) == len(MODELS):
            label = "All 4 correct"
        else:
            label = " + ".join(MODEL_LABELS[m] for m in MODELS if m in combo)
        combo_table.append({
            "combination": label,
            "models":      sorted(list(combo)),
            "count":       count,
            "pct":         round(count / n_valid * 100, 1),
        })

    # ── n_correct distribution ─────────────────────────────────────────────────
    n_correct_dist = {
        k: {"count": v, "pct": round(v / n_valid * 100, 1)}
        for k, v in sorted(n_correct_counts.items())
    }

    # ── Model win rates ────────────────────────────────────────────────────────
    model_stats = {}
    for model in MODELS:
        model_stats[model] = {
            "clips_correct":  model_correct[model],
            "pct_correct":    round(model_correct[model] / n_valid * 100, 1),
            "clips_best_wer": model_win_counts[model],
            "pct_best_wer":   round(model_win_counts[model] / n_valid * 100, 1),
        }

    # ── Complementarity score ──────────────────────────────────────────────────
    # % of clips where the best model differs from Qwen3-ASR (our best overall)
    # High score = models are complementary = combination is valuable
    qwen_wins   = model_win_counts["qwen"]
    others_win  = n_valid - qwen_wins
    complementarity = round(others_win / n_valid * 100, 1)

    return {
        "dataset":            dataset,
        "threshold":          threshold,
        "n_clips":            n_valid,
        "combination_table":  combo_table,
        "n_correct_dist":     n_correct_dist,
        "model_stats":        model_stats,
        "complementarity_pct": complementarity,
    }

# ── Plotting ───────────────────────────────────────────────────────────────────

def make_plots(results: dict, dataset: str, plots_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    # ── Plot 1: Distribution — how many models correct per clip ───────────────
    dist = results["n_correct_dist"]
    x_labels = [f"{k} model{'s' if k != 1 else ''} correct" for k in sorted(dist.keys())]
    counts   = [dist[k]["count"] for k in sorted(dist.keys())]
    pcts     = [dist[k]["pct"]   for k in sorted(dist.keys())]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x_labels, counts, color=colours[:len(x_labels)])
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, f"{pct}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of clips")
    ax.set_title(f"How many models correct per clip — {dataset}\n(threshold WER ≤ {results['threshold']})")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"agreement_{dataset}_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 2: Top combinations bar chart ────────────────────────────────────
    combo_table = results["combination_table"][:12]  # top 12
    labels  = [r["combination"] for r in combo_table]
    counts  = [r["count"]       for r in combo_table]
    pcts    = [r["pct"]         for r in combo_table]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels[::-1], counts[::-1], color="#4C72B0")
    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=9)
    ax.set_xlabel("Number of clips")
    ax.set_title(f"Most common model agreement combinations — {dataset}\n(threshold WER ≤ {results['threshold']})")
    ax.set_xlim(0, max(counts) * 1.2)
    plt.tight_layout()
    path = os.path.join(plots_dir, f"agreement_{dataset}_combinations.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Plot 3: Per-model win rate (% of clips each model has best WER) ───────
    model_stats = results["model_stats"]
    models  = MODELS
    correct = [model_stats[m]["pct_correct"]  for m in models]
    best    = [model_stats[m]["pct_best_wer"] for m in models]
    labels  = [MODEL_LABELS[m] for m in models]

    import numpy as np
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, correct, width, label="% clips correct", color="#4C72B0")
    b2 = ax.bar(x + width/2, best,    width, label="% clips best WER", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of clips")
    ax.set_title(f"Per-model performance — {dataset}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(plots_dir, f"agreement_{dataset}_model_wins.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="all")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="WER threshold for 'correct' (default 0.2)")
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    threshold_dir = os.path.join(OUTPUT_DIR, f"threshold_{args.threshold}")
    os.makedirs(threshold_dir, exist_ok=True)
    plots_dir = os.path.join(threshold_dir, "plots")
    if not args.no_plots:
        os.makedirs(plots_dir, exist_ok=True)

    for dataset in datasets:
        results = analyse_dataset(dataset, args.threshold)

        # save JSON
        json_path = os.path.join(threshold_dir, f"agreement_{dataset}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {json_path}")
        print(f"  Output dir: {threshold_dir}")

        # print summary
        print(f"\n  ── {dataset} summary ──")
        print(f"  Complementarity: {results['complementarity_pct']}% of clips won by non-Qwen model")
        print(f"\n  n correct | count | %")
        for k, v in sorted(results["n_correct_dist"].items()):
            print(f"  {k} models   | {v['count']:>5} | {v['pct']:>5}%")

        print(f"\n  Top combinations:")
        for row in results["combination_table"][:8]:
            print(f"    {row['combination']:<45} {row['count']:>5} ({row['pct']}%)")

        print(f"\n  Model stats:")
        print(f"  {'Model':<12} {'Correct%':>10} {'Best WER%':>10}")
        for m in MODELS:
            s = results["model_stats"][m]
            print(f"  {m:<12} {s['pct_correct']:>9}% {s['pct_best_wer']:>9}%")

        if not args.no_plots:
            print(f"\n  Generating plots...")
            make_plots(results, dataset, plots_dir)

    print(f"\nDone. Outputs in {threshold_dir}/")

if __name__ == "__main__":
    main()