"""
calibration_summary.py

Reads human_eval_samples.json and outputs agreement, precision, recall, F1,
FN rate, and FP rate for every judge/prompt combination found in the file.

Usage:
    python scripts/calibration_summary.py
    python scripts/calibration_summary.py --input human_eval_samples.json
"""

import json
import argparse

KNOWN_JUDGES = [
    "gpt4o_verdict",
    "gpt4o_verdict_p2",
    "gpt4o_verdict_p3",
    "selene_verdict",
    "selene_verdict_p2",
    "selene_verdict_p3",
    "qwen_verdict",
    "qwen_verdict_p2",
    "qwen_verdict_p3",
    "llama_verdict",
    "llama_verdict_p2",
    "llama_verdict_p3",
    "llama4_verdict_p1",
    "llama4_verdict_p2",
    "llama4_verdict_p3",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="human_eval_samples.json")
    args = parser.parse_args()

    with open(args.input) as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from {args.input}\n")

    # find which judge fields actually exist in the file
    all_fields = set()
    for s in samples:
        all_fields.update(s.keys())

    judges = [j for j in KNOWN_JUDGES if j in all_fields]
    extra  = [k for k in all_fields if "verdict" in k and k not in KNOWN_JUDGES and k != "human_verdict"]
    judges += extra

    if not judges:
        print("No judge verdict fields found.")
        return

    valid = [s for s in samples if s.get("human_verdict") is not None]
    n_pos = sum(1 for s in valid if s["human_verdict"] is True)
    n_neg = sum(1 for s in valid if s["human_verdict"] is False)

    print(f"Samples with human verdict: {len(valid)} ({n_pos} positive, {n_neg} negative)\n")

    header = f"{'Judge':<25} {'N':>6} {'Agree':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FN':>7} {'FP':>7}"
    print(header)
    print("-" * 80)

    for judge in judges:
        judged = [s for s in valid if s.get(judge) is not None]
        if not judged:
            print(f"{judge:<25} {'—':>6}")
            continue

        n   = len(judged)
        tp  = sum(1 for s in judged if s["human_verdict"] is True  and s[judge] is True)
        tn  = sum(1 for s in judged if s["human_verdict"] is False and s[judge] is False)
        fp  = sum(1 for s in judged if s["human_verdict"] is False and s[judge] is True)
        fn  = sum(1 for s in judged if s["human_verdict"] is True  and s[judge] is False)

        pos = sum(1 for s in judged if s["human_verdict"] is True)
        neg = sum(1 for s in judged if s["human_verdict"] is False)

        agree     = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fn_rate   = fn / pos if pos > 0 else 0
        fp_rate   = fp / neg if neg > 0 else 0

        print(f"{judge:<25} {n:>6} "
              f"{agree*100:>6.1f}% "
              f"{precision*100:>6.1f}% "
              f"{recall*100:>6.1f}% "
              f"{f1*100:>6.1f}% "
              f"{fn_rate*100:>6.1f}% "
              f"{fp_rate*100:>6.1f}%")

    print()

if __name__ == "__main__":
    main()