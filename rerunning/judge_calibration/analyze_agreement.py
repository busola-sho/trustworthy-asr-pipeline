"""
rerunning/judge_calibration/analyze_agreement.py

Computes agreement between each judge's severity scores and human_severity
labels, across DIRECT, MEDIUM, and STRUCTURED prompt conditions. Reports:
  - QWK (primary metric - rewards near-misses, penalizes big misses)
  - Exact-match accuracy (secondary sanity check)
  - Mean absolute error (average severity levels off, by direction-agnostic distance)
  - Confusion matrix (which levels get confused with which)
  - Binary MAR F1/recall/precision (0 = no meaning change vs 1-4 = meaning
    change collapsed) - recall specifically flags whether severe errors are
    being missed, which matters most for the Police Scotland use case
  - Format-failure rate (rows where the judge's response couldn't be parsed)

Gracefully skips any judge whose result file doesn't exist yet (e.g. if
DeepSeek hasn't been run) rather than failing - prints a note and continues
with whatever is available.

Usage:
    python rerunning/judge_calibration/analyze_agreement.py
    python rerunning/judge_calibration/analyze_agreement.py --results-dir writeup_results/judge_calibration
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, mean_absolute_error,
    confusion_matrix, f1_score, recall_score, precision_score,
)

RESULTS_DIR = "writeup_results/judge_calibration"

ALL_JUDGES = ["qwen2.5", "qwen3.5", "gemma4", "phi4", "ministral3", "deepseek", "gpt5.6luna"]
CONDITIONS = ["direct", "medium", "structured"]
SEVERITY_LABELS = [0, 1, 2, 3, 4]


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def binarize_mar(severities):
    """0 stays 'no meaning change'; 1-4 collapse to 'meaning change'."""
    return [0 if s == 0 else 1 for s in severities]


def compute_agreement(results: list):
    """
    Returns a dict of all reported metrics for one judge/condition result
    set, or None if there are no valid rows. Only rows with both a valid
    human_severity and a valid judge severity are included in the metric
    comparison - excluded/unparseable rows are counted separately as the
    format-failure rate.
    """
    n_total = len(results)
    valid = [
        r for r in results
        if r.get("human_severity") is not None and r.get("severity") is not None
    ]
    n_valid = len(valid)
    n_failed = n_total - n_valid

    if n_valid == 0:
        return None

    human = [r["human_severity"] for r in valid]
    judge = [r["severity"] for r in valid]

    qwk = cohen_kappa_score(human, judge, weights="quadratic")
    exact = accuracy_score(human, judge) * 100
    mae = mean_absolute_error(human, judge)
    cm = confusion_matrix(human, judge, labels=SEVERITY_LABELS)

    human_bin = binarize_mar(human)
    judge_bin = binarize_mar(judge)
    mar_f1 = f1_score(human_bin, judge_bin, zero_division=0)
    mar_recall = recall_score(human_bin, judge_bin, zero_division=0)
    mar_precision = precision_score(human_bin, judge_bin, zero_division=0)

    return {
        "qwk": qwk,
        "exact_match_pct": exact,
        "mae": mae,
        "confusion_matrix": cm.tolist(),
        "mar_f1": mar_f1,
        "mar_recall": mar_recall,
        "mar_precision": mar_precision,
        "n_valid": n_valid,
        "n_total": n_total,
        "n_failed": n_failed,
        "format_failure_rate_pct": (n_failed / n_total * 100) if n_total else 0,
    }


def plot_confusion_matrix(cm, judge, condition, qwk, rank, output_dir: Path):
    """Saves a blue-gradient heatmap confusion matrix PNG, dissertation-ready."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(SEVERITY_LABELS)))
    ax.set_yticks(range(len(SEVERITY_LABELS)))
    ax.set_xticklabels(SEVERITY_LABELS)
    ax.set_yticklabels(SEVERITY_LABELS)
    ax.set_xlabel("Judge severity")
    ax.set_ylabel("Human severity")
    ax.set_title(f"#{rank}: {judge} / {condition}  (QWK={qwk:.3f})")

    # annotate each cell with its count; text color flips for contrast
    # against the darker cells in the middle/high end of the colormap
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > vmax * 0.6 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"confusion_{rank}_{judge}_{condition}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_confusion_matrix(cm, judge, condition):
    print(f"\n  (rows = your human severity, columns = judge's severity)")
    header = "         " + "".join(f"{c:>6}" for c in SEVERITY_LABELS)
    print(header)
    for i, row in enumerate(cm):
        print(f"  human={SEVERITY_LABELS[i]:>2} |" + "".join(f"{v:>6}" for v in row))


def print_condition_table(condition, rows):
    """One table per prompt condition, all judges, best row starred."""
    if not rows:
        return
    best_qwk = max(r["qwk"] for r in rows)

    print(f"\n{'='*70}")
    print(f"  {condition.upper()}")
    print(f"{'='*70}")
    header = f"{'Judge':<12} {'QWK':>6} {'Exact%':>7} {'MAE':>5} {'MAR-Rec':>8} {'Fail%':>6}"
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda r: -r["qwk"]):
        star = " ★ BEST" if r["qwk"] == best_qwk else ""
        print(f"{r['judge']:<12} {r['qwk']:>6.3f} {r['exact_match_pct']:>6.1f}% "
              f"{r['mae']:>5.2f} {r['mar_recall']:>8.3f} {r['format_failure_rate_pct']:>5.1f}%{star}")


def print_best_overall_table(summary_rows):
    """One row per judge: its single best condition, across all three."""
    by_judge = {}
    for row in summary_rows:
        j = row["judge"]
        if j not in by_judge or row["qwk"] > by_judge[j]["qwk"]:
            by_judge[j] = row

    best_rows = sorted(by_judge.values(), key=lambda r: -r["qwk"])
    overall_best_qwk = best_rows[0]["qwk"] if best_rows else None

    print(f"\n{'='*70}")
    print(f"  BEST OVERALL (best condition per judge)")
    print(f"{'='*70}")
    header = f"{'Judge':<12} {'Best cond':<11} {'QWK':>6} {'Exact%':>7} {'MAE':>5} {'MAR-Rec':>8}"
    print(header)
    print("-" * len(header))
    for r in best_rows:
        star = " ★ OVERALL BEST" if r["qwk"] == overall_best_qwk else ""
        print(f"{r['judge']:<12} {r['condition']:<11} {r['qwk']:>6.3f} "
              f"{r['exact_match_pct']:>6.1f}% {r['mae']:>5.2f} {r['mar_recall']:>8.3f}{star}")

    return best_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--judges", nargs="+", default=ALL_JUDGES)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    summary_rows = []
    missing = []
    rows_by_condition = {c: [] for c in CONDITIONS}

    for judge in args.judges:
        for condition in CONDITIONS:
            file_path = results_dir / f"{judge}_{condition}.json"

            if not file_path.exists():
                missing.append(str(file_path))
                continue

            results = load_results(file_path)
            m = compute_agreement(results)

            if m is None:
                continue

            row = {
                "judge": judge,
                "condition": condition,
                "qwk": round(m["qwk"], 4),
                "exact_match_pct": round(m["exact_match_pct"], 2),
                "mae": round(m["mae"], 4),
                "mar_f1": round(m["mar_f1"], 4),
                "mar_recall": round(m["mar_recall"], 4),
                "mar_precision": round(m["mar_precision"], 4),
                "format_failure_rate_pct": round(m["format_failure_rate_pct"], 2),
                "confusion_matrix": m["confusion_matrix"],
                "n_valid": m["n_valid"],
                "n_total": m["n_total"],
                "n_failed": m["n_failed"],
            }
            summary_rows.append(row)
            rows_by_condition[condition].append(row)

    # ── Three per-condition tables ──
    for condition in CONDITIONS:
        print_condition_table(condition, rows_by_condition[condition])

    # ── One best-overall table ──
    best_per_judge = print_best_overall_table(summary_rows)

    # ── Colour heatmaps for the top 3 UNIQUE judges (best condition each), ──
    # ── dissertation-ready - never the same judge twice ──
    top3 = sorted(best_per_judge, key=lambda r: -r["qwk"])[:3]
    if top3:
        confusion_dir = results_dir / "confusion_matrices"
        print(f"\n{'='*70}")
        print(f"  TOP 3 CONFUSION MATRICES (saved as PNG heatmaps)")
        print(f"{'='*70}")
        for rank, row in enumerate(top3, start=1):
            out_path = plot_confusion_matrix(
                row["confusion_matrix"], row["judge"], row["condition"],
                row["qwk"], rank, confusion_dir,
            )
            print(f"  #{rank}: {row['judge']} / {row['condition']} (QWK={row['qwk']:.3f}) -> {out_path}")

    if missing:
        print(f"\n({len(missing)} file(s) not yet run, skipped)")

    # save the full summary (including all confusion matrices) for the writeup
    summary_path = results_dir / "agreement_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nFull summary (all judges/conditions/confusion matrices) saved to {summary_path}")

    print("\nNote on efficiency (runtime/API cost): not captured by this script since")
    print("run_judge_calibration.py doesn't currently log per-call timing/token usage.")
    print("Known from manual testing: Phi-4 ~10s/call, Ministral-3/Gemma4 ~10-35s/call,")
    print("DeepSeek + GPT-5.6 Luna API calls are fast and together cost under $4 for the")
    print("full 100-sentence x 3-condition run.")


if __name__ == "__main__":
    main()