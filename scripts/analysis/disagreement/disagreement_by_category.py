"""
disagreement_by_category.py

For each pair of models, when they disagree on a word, what category is that
word and who is right?

For each pair (A, B) and each word category, counts:
- A right, B wrong  — A wins
- B right, A wrong  — B wins
- Both wrong        — neither saves the word
- Both right        — shouldn't happen if they disagree, sanity check

Outputs:
    analysis/disagreement_by_category.json
    analysis/plots/disagreement_category_*.png

Usage:
    python scripts/disagreement_by_category.py
    python scripts/disagreement_by_category.py --dataset commonvoice
    python scripts/disagreement_by_category.py --no-plots
"""

import json
import os
import argparse
from itertools import combinations
from collections import defaultdict

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

MODELS   = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

MODEL_LABELS = {
    "qwen":     "Qwen3-ASR",
    "whisper":  "Whisper",
    "parakeet": "Parakeet",
    "wav2vec2": "wav2vec2",
}

CATEGORIES = ["dialect", "number", "named_entity", "function", "content"]

# ── Word category ──────────────────────────────────────────────────────────────

DIALECT_WORDS = {
    "weans", "wean", "oot", "aboot", "didnae", "wasnae", "cannae", "dinnae",
    "wouldnae", "couldnae", "shouldnae", "havenae", "isnae", "doesnae",
    "maist", "braw", "aye", "nae", "fae", "wi", "tae", "wee", "bide",
    "minted", "quine", "quines", "loon", "tatties", "neep", "cooncil",
    "ootside", "masel", "mysel", "roond", "doon", "heid", "guid", "crabbit",
    "muckle", "lang", "athing", "alang", "fann", "farr", "fitt",
}

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "not", "no", "so", "as", "if", "then", "than", "when", "where",
    "which", "who", "what", "how", "there", "here", "just", "also",
}

def word_category(w: str) -> str:
    clean = w.lower().strip(".,!?;:\"'()-[]")
    if clean in DIALECT_WORDS:
        return "dialect"
    if clean.isdigit():
        return "number"
    if w[0].isupper() and len(w) > 1 and not w.isupper():
        return "named_entity"
    if clean in FUNCTION_WORDS:
        return "function"
    return "content"

# ── Alignment ──────────────────────────────────────────────────────────────────

def get_aligned_output(ref: str, hyp: str):
    """
    Returns list of (ref_word, hyp_word_or_None, is_correct) per ref position.
    """
    try:
        out = process_words(ref, hyp)
        ref_words = out.references[0]
        hyp_words = out.hypotheses[0]
        alignments = out.alignments[0]

        result = []
        for _ in ref_words:
            result.append({"ref": None, "hyp": None, "correct": False})

        for i, w in enumerate(ref_words):
            result[i]["ref"] = w

        for op in alignments:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i]["hyp"]     = hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None
                    result[i]["correct"] = True
            elif op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i]["hyp"]     = hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None
                    result[i]["correct"] = False
            elif op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    result[i]["hyp"]     = None
                    result[i]["correct"] = False

        return result
    except Exception:
        return []

# ── Core analysis ──────────────────────────────────────────────────────────────

def analyse_dataset(dataset: str) -> dict:
    print(f"\n  Analysing {dataset}...")

    model_samples = {}
    for model in MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            data = json.load(f)
        model_samples[model] = data["samples"]

    n = len(model_samples["qwen"])

    # for each pair and category: {a_wins, b_wins, both_wrong, both_right}
    # structure: pair -> category -> outcome -> count
    pair_cat_counts = {
        (a, b): {cat: defaultdict(int) for cat in CATEGORIES}
        for a, b in combinations(MODELS, 2)
    }

    n_valid = 0

    for i in range(n):
        ref = model_samples["qwen"][i]["ref"]
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        n_valid += 1

        # get alignment per model
        alignments = {}
        for model in MODELS:
            hyp = model_samples[model][i]["hyp"]
            alignments[model] = get_aligned_output(ref, hyp)

        ref_len = len(alignments["qwen"])
        if ref_len == 0:
            continue

        for j in range(ref_len):
            ref_word = alignments["qwen"][j]["ref"]
            if not ref_word:
                continue

            cat = word_category(ref_word)

            # correctness per model at this position
            correct = {}
            for model in MODELS:
                if j < len(alignments[model]):
                    correct[model] = alignments[model][j]["correct"]
                else:
                    correct[model] = False

            # compare each pair
            for a, b in combinations(MODELS, 2):
                a_correct = correct[a]
                b_correct = correct[b]

                # only count disagreements
                if a_correct == b_correct:
                    if a_correct:
                        pair_cat_counts[(a, b)][cat]["both_right"] += 1
                    else:
                        pair_cat_counts[(a, b)][cat]["both_wrong"] += 1
                else:
                    if a_correct:
                        pair_cat_counts[(a, b)][cat]["a_wins"] += 1
                    else:
                        pair_cat_counts[(a, b)][cat]["b_wins"] += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} samples")

    print(f"    {n_valid} valid samples")

    # build results
    results = {}
    for a, b in combinations(MODELS, 2):
        pair_key = f"{a}_vs_{b}"
        results[pair_key] = {
            "model_a": a,
            "model_b": b,
            "label_a": MODEL_LABELS[a],
            "label_b": MODEL_LABELS[b],
            "by_category": {}
        }

        for cat in CATEGORIES:
            counts = pair_cat_counts[(a, b)][cat]
            a_wins     = counts.get("a_wins",     0)
            b_wins     = counts.get("b_wins",     0)
            both_wrong = counts.get("both_wrong", 0)
            both_right = counts.get("both_right", 0)
            total_disagree = a_wins + b_wins

            results[pair_key]["by_category"][cat] = {
                f"{a}_wins":   a_wins,
                f"{b}_wins":   b_wins,
                "both_wrong":  both_wrong,
                "both_right":  both_right,
                "total_disagree": total_disagree,
                f"{a}_win_rate": round(a_wins / total_disagree, 3) if total_disagree else 0,
                f"{b}_win_rate": round(b_wins / total_disagree, 3) if total_disagree else 0,
                "both_wrong_rate": round(both_wrong / (both_wrong + total_disagree), 3)
                                   if (both_wrong + total_disagree) else 0,
            }

    return {
        "dataset": dataset,
        "n_valid": n_valid,
        "pairs":   results,
    }

# ── Plotting ───────────────────────────────────────────────────────────────────

def make_plots(results: dict, dataset: str, plots_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import numpy as np
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    pairs = results["pairs"]
    cat_labels = {
        "dialect":      "Dialect",
        "number":       "Number",
        "named_entity": "Named Entity",
        "function":     "Function",
        "content":      "Content",
    }

    # one plot per pair showing category breakdown
    for pair_key, pair_data in pairs.items():
        a = pair_data["model_a"]
        b = pair_data["model_b"]
        la = MODEL_LABELS[a]
        lb = MODEL_LABELS[b]

        cats = CATEGORIES
        a_win_rates  = []
        b_win_rates  = []
        bw_rates     = []
        n_disagrees  = []

        for cat in cats:
            cd = pair_data["by_category"][cat]
            total = cd["total_disagree"]
            n_disagrees.append(total)
            if total > 0:
                a_win_rates.append(cd.get(f"{a}_win_rate", 0) * 100)
                b_win_rates.append(cd.get(f"{b}_win_rate", 0) * 100)
                bw_rates.append(cd.get("both_wrong_rate", 0) * 100)
            else:
                a_win_rates.append(0)
                b_win_rates.append(0)
                bw_rates.append(0)

        x = np.arange(len(cats))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        b1 = ax.bar(x - width, a_win_rates, width, label=f"{la} wins", color="#4C72B0")
        b2 = ax.bar(x,         b_win_rates, width, label=f"{lb} wins", color="#DD8452")
        b3 = ax.bar(x + width, bw_rates,    width, label="Both wrong", color="#C44E52", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{cat_labels[c]}\n(n={n_disagrees[i]})"
                            for i, c in enumerate(cats)], fontsize=9)
        ax.set_ylabel("% of disagreements")
        ax.set_ylim(0, 110)
        ax.set_title(f"When {la} and {lb} disagree — who is right?\n{dataset}")
        ax.legend()
        plt.tight_layout()

        path = os.path.join(plots_dir, f"disagree_{dataset}_{a}_vs_{b}.png")
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

    for dataset in datasets:
        results = analyse_dataset(dataset)

        # print summary
        print(f"\n  ── {dataset} disagreement by category ──")
        for pair_key, pair_data in results["pairs"].items():
            a  = pair_data["model_a"]
            b  = pair_data["model_b"]
            la = MODEL_LABELS[a]
            lb = MODEL_LABELS[b]
            print(f"\n  {la} vs {lb}:")
            print(f"  {'Category':<14} {'Disagree n':>11} {la+' wins':>12} {lb+' wins':>12} {'Both wrong':>11}")
            for cat in CATEGORIES:
                cd    = pair_data["by_category"][cat]
                total = cd["total_disagree"]
                if total == 0:
                    continue
                a_rate = cd.get(f"{a}_win_rate", 0) * 100
                b_rate = cd.get(f"{b}_win_rate", 0) * 100
                bw     = cd.get("both_wrong_rate", 0) * 100
                print(f"  {cat:<14} {total:>11} {a_rate:>11.1f}% {b_rate:>11.1f}% {bw:>10.1f}%")

        # save JSON
        out_path = os.path.join(OUTPUT_DIR, f"disagreement_category_{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

        if not args.no_plots:
            print(f"\n  Generating plots...")
            make_plots(results, dataset, plots_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()