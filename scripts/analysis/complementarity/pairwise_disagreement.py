"""
pairwise_disagreement.py

Computes pairwise disagreement rates between all 6 model pairs across datasets.

For each pair of models (A, B), counts the fraction of word positions where
they produce different outputs. High disagreement = high complementarity =
more to gain from combination.

Outputs:
    analysis/pairwise_disagreement.json  — full results
    analysis/plots/pairwise_*.png        — heatmaps per dataset

Usage:
    python scripts/pairwise_disagreement.py
    python scripts/pairwise_disagreement.py --dataset commonvoice
    python scripts/pairwise_disagreement.py --no-plots
"""

import json
import os
import argparse
from itertools import combinations

from jiwer import process_words

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

MODELS = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

MODEL_LABELS = {
    "qwen":     "Qwen3-ASR",
    "whisper":  "Whisper",
    "parakeet": "Parakeet",
    "wav2vec2": "wav2vec2",
}

# ── Core ───────────────────────────────────────────────────────────────────────

def get_hypothesis_words(ref: str, hyp: str) -> list:
    """
    Align hyp to ref and return a list of what the model produced
    at each reference word position. Returns None for deletions.
    """
    try:
        out = process_words(ref, hyp)
        ref_words = out.references[0]
        hyp_words = out.hypotheses[0]
        alignments = out.alignments[0]

        # build a mapping: ref_position -> what model produced
        result = [None] * len(ref_words)

        for op in alignments:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None
            elif op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None
            elif op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    result[i] = None  # deletion — model produced nothing here

        return result
    except Exception:
        return []

def analyse_dataset(dataset: str) -> dict:
    print(f"\n  Analysing {dataset}...")

    # load all 4 model files
    model_samples = {}
    for model in MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            data = json.load(f)
        model_samples[model] = data["samples"]

    n = len(model_samples["qwen"])

    # pairwise disagreement counts
    pair_disagree = {(a, b): 0 for a, b in combinations(MODELS, 2)}
    pair_total    = {(a, b): 0 for a, b in combinations(MODELS, 2)}

    n_valid = 0

    for i in range(n):
        ref = model_samples["qwen"][i]["ref"]
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        n_valid += 1

        # get what each model produced at each reference word position
        model_outputs = {}
        for model in MODELS:
            hyp = model_samples[model][i]["hyp"]
            model_outputs[model] = get_hypothesis_words(ref, hyp)

        # get reference length from qwen alignment
        ref_len = len(model_outputs["qwen"])
        if ref_len == 0:
            continue

        # compare each pair at each word position
        for a, b in combinations(MODELS, 2):
            out_a = model_outputs[a]
            out_b = model_outputs[b]
            compare_len = min(len(out_a), len(out_b), ref_len)

            for j in range(compare_len):
                pair_total[(a, b)] += 1
                if out_a[j] != out_b[j]:
                    pair_disagree[(a, b)] += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} samples")

    print(f"    {n_valid} valid samples")

    # compute disagreement rates
    disagreement_rates = {}
    for a, b in combinations(MODELS, 2):
        total = pair_total[(a, b)]
        disagree = pair_disagree[(a, b)]
        rate = round(disagree / total, 4) if total else 0
        disagreement_rates[f"{a}_vs_{b}"] = {
            "model_a":          a,
            "model_b":          b,
            "disagreement_rate": rate,
            "disagreements":    disagree,
            "total_positions":  total,
        }

    # build full 4x4 matrix (symmetric, diagonal = 0)
    matrix = {a: {b: 0.0 for b in MODELS} for a in MODELS}
    for a, b in combinations(MODELS, 2):
        rate = disagreement_rates[f"{a}_vs_{b}"]["disagreement_rate"]
        matrix[a][b] = rate
        matrix[b][a] = rate  # symmetric

    # rank pairs by disagreement rate
    ranked_pairs = sorted(
        disagreement_rates.items(),
        key=lambda x: x[1]["disagreement_rate"],
        reverse=True
    )

    return {
        "dataset":            dataset,
        "n_valid":            n_valid,
        "disagreement_rates": disagreement_rates,
        "matrix":             matrix,
        "ranked_pairs":       [(k, v["disagreement_rate"]) for k, v in ranked_pairs],
    }

# ── Plotting ───────────────────────────────────────────────────────────────────

def make_heatmap(results: dict, dataset: str, plots_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    matrix = results["matrix"]
    labels = [MODEL_LABELS[m] for m in MODELS]
    data   = np.array([[matrix[a][b] for b in MODELS] for a in MODELS])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            val = data[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Disagreement rate")
    ax.set_title(f"Pairwise Disagreement Rate — {dataset}\n"
                 f"(higher = more complementary)")
    plt.tight_layout()

    path = os.path.join(plots_dir, f"pairwise_{dataset}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="all")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    if not args.no_plots:
        os.makedirs(plots_dir, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        results = analyse_dataset(dataset)
        all_results[dataset] = results

        # print results
        print(f"\n  ── {dataset} pairwise disagreement ──")
        print(f"  {'Pair':<35} {'Disagreement':>14}")
        for pair, rate in results["ranked_pairs"]:
            a, b = pair.split("_vs_")
            label = f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]}"
            print(f"  {label:<35} {rate*100:>13.1f}%")

        # save JSON
        out_path = os.path.join(OUTPUT_DIR, f"pairwise_{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

        if not args.no_plots:
            make_heatmap(results, dataset, plots_dir)

    # cross-dataset summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY — most complementary pairs")
    print(f"{'='*60}")
    for dataset, results in all_results.items():
        top_pair, top_rate = results["ranked_pairs"][0]
        a, b = top_pair.split("_vs_")
        print(f"  {dataset:<22} most complementary: "
              f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]} ({top_rate*100:.1f}%)")

    print("\nDone.")

if __name__ == "__main__":
    main()