"""
sample_complementarity_cases.py

Samples cases from each complementarity category for manual qualitative analysis:
- both_fix: both Whisper and Parakeet avoid Qwen's MA error
- whisper_only: only Whisper avoids Qwen's MA error
- parakeet_only: only Parakeet avoids Qwen's MA error
- neither: neither model avoids Qwen's MA error

Outputs a readable text file for manual annotation.

Usage:
    python scripts/sample_complementarity_cases.py --dataset commonvoice --n 20
    python scripts/sample_complementarity_cases.py --dataset commonvoice --n 10 --output analysis/qualitative_commonvoice.txt
"""

import json
import os
import random
import argparse

BENCHMARKS_DIR = "benchmarks"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "qwen_english_dialects_20260525_000627.json",
    ("qwen",     "shetland"):         "shetland_qwen3asr_20260603_150124.json",
    ("whisper",  "commonvoice"):      "whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_20260525_110315.json",
    ("whisper",  "shetland"):         "shetland_whisper_20260603_123115.json",
    ("parakeet", "commonvoice"):      "parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_20260524_234807.json",
    ("parakeet", "shetland"):         "shetland_parakeet_20260606_134131.json",
    ("wav2vec2", "commonvoice"):      "wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "wav2vec2_english_dialects_20260526_073439.json",
    ("wav2vec2", "shetland"):         "shetland_wav2vec2_20260606_134507.json",
}

CATEGORIES = ["both_fix", "whisper_only", "parakeet_only", "neither"]

def categorise(q_verdict, w_verdict, p_verdict):
    q_ma = q_verdict is True
    w_ok = w_verdict is False
    p_ok = p_verdict is False

    if not q_ma:
        return None  # Qwen has no MA error — not relevant

    if w_ok and p_ok:
        return "both_fix"
    elif w_ok and not p_ok:
        return "whisper_only"
    elif p_ok and not w_ok:
        return "parakeet_only"
    else:
        return "neither"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice",
                        choices=["commonvoice", "edacc", "english_dialects", "shetland"])
    parser.add_argument("--n",       type=int, default=20,
                        help="Number of samples per category (default: 20)")
    parser.add_argument("--output",  default=None,
                        help="Output file path (default: analysis/qualitative_{dataset}.txt)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # load model samples
    model_samples = {}
    for model in ["qwen", "whisper", "parakeet", "wav2vec2"]:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, args.dataset)])
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    n = len(model_samples["qwen"])

    # bucket all samples by category
    buckets = {cat: [] for cat in CATEGORIES}

    for i in range(n):
        ref = model_samples["qwen"][i]["ref"]
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        q_verdict = model_samples["qwen"][i].get("qwen_verdict_p2")
        w_verdict = model_samples["whisper"][i].get("qwen_verdict_p2")
        p_verdict = model_samples["parakeet"][i].get("qwen_verdict_p2")

        if any(v is None for v in [q_verdict, w_verdict, p_verdict]):
            continue

        cat = categorise(q_verdict, w_verdict, p_verdict)
        if cat is None:
            continue

        buckets[cat].append(i)

    # sample from each bucket
    sampled = {
        cat: random.sample(indices, min(args.n, len(indices)))
        for cat, indices in buckets.items()
    }

    # output path
    os.makedirs("analysis", exist_ok=True)
    out_path = args.output or f"analysis/qualitative_{args.dataset}.txt"

    CATEGORY_LABELS = {
        "both_fix":     "BOTH WHISPER AND PARAKEET FIX QWEN'S ERROR",
        "whisper_only": "WHISPER ONLY FIXES QWEN'S ERROR",
        "parakeet_only":"PARAKEET ONLY FIXES QWEN'S ERROR",
        "neither":      "NEITHER MODEL FIXES QWEN'S ERROR",
    }

    with open(out_path, "w") as f:
        f.write(f"Qualitative Complementarity Analysis — {args.dataset}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Bucket sizes:\n")
        for cat in CATEGORIES:
            f.write(f"  {cat:<15} total={len(buckets[cat]):<6} sampled={len(sampled[cat])}\n")
        f.write(f"\n{'='*70}\n\n")

        for cat in CATEGORIES:
            f.write(f"\n{'='*70}\n")
            f.write(f"CATEGORY: {CATEGORY_LABELS[cat]}\n")
            f.write(f"{'='*70}\n\n")

            for rank, i in enumerate(sampled[cat], 1):
                ref     = model_samples["qwen"][i]["ref"]
                q_hyp   = model_samples["qwen"][i]["hyp"]
                w_hyp   = model_samples["whisper"][i]["hyp"]
                p_hyp   = model_samples["parakeet"][i]["hyp"]
                w2_hyp  = model_samples["wav2vec2"][i]["hyp"]
                q_wer   = model_samples["qwen"][i].get("sample_WER", "?")
                w_wer   = model_samples["whisper"][i].get("sample_WER", "?")
                p_wer   = model_samples["parakeet"][i].get("sample_WER", "?")

                f.write(f"── Sample {rank} (clip index {i}) ──────────────────────────────\n")
                f.write(f"REF:      {ref}\n\n")
                f.write(f"QWEN:     {q_hyp}\n")
                f.write(f"          WER={q_wer:.3f}  MA=true\n\n")
                f.write(f"WHISPER:  {w_hyp}\n")
                f.write(f"          WER={w_wer:.3f}  MA={model_samples['whisper'][i].get('qwen_verdict_p2')}\n\n")
                f.write(f"PARAKEET: {p_hyp}\n")
                f.write(f"          WER={p_wer:.3f}  MA={model_samples['parakeet'][i].get('qwen_verdict_p2')}\n\n")
                f.write(f"ERROR TYPE (fill in): ___________________________\n")
                f.write(f"NOTES:               ___________________________\n")
                f.write(f"\n")

    print(f"Saved {sum(len(v) for v in sampled.values())} samples to {out_path}")
    print(f"\nBucket summary:")
    for cat in CATEGORIES:
        print(f"  {cat:<15} total={len(buckets[cat]):<6} sampled={len(sampled[cat])}")

if __name__ == "__main__":
    main()