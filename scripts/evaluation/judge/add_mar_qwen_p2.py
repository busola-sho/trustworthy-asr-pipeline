"""
add_mar_qwen_p2.py

Computes meaning_alteration_rate_qwen_p2 from existing qwen_verdict_p2
per-sample fields and adds it to the JSON header. No LLM calls needed.

Usage:
    python scripts/add_mar_qwen_p2.py --model qwen --dataset commonvoice
    python scripts/add_mar_qwen_p2.py --all
"""

import json
import os
import argparse

BENCHMARKS_DIR = "benchmarks"

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

def process_file(path):
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    valid   = [s for s in samples
               if not s.get("skipped") and not s.get("error")
               and s.get("qwen_verdict_p2") is not None]

    if not valid:
        return None, 0, len(samples), None

    mar        = sum(1 for s in valid if s["qwen_verdict_p2"]) / len(valid)
    corpus_wer = data.get("corpus_wer")

    data["meaning_alteration_rate_qwen_p2"] = mar

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return mar, len(valid), len(samples), corpus_wer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None,
                        choices=["qwen", "whisper", "parakeet", "wav2vec2"])
    parser.add_argument("--dataset", default=None,
                        choices=["commonvoice", "edacc", "english_dialects"])
    parser.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    if args.all:
        targets = list(CANONICAL_FILES.keys())
    elif args.model and args.dataset:
        targets = [(args.model, args.dataset)]
    elif args.model:
        targets = [(args.model, ds) for ds in ["commonvoice", "edacc", "english_dialects"]
                   if (args.model, ds) in CANONICAL_FILES]
    elif args.dataset:
        targets = [(m, args.dataset) for m in ["qwen", "whisper", "parakeet", "wav2vec2"]
                   if (m, args.dataset) in CANONICAL_FILES]
    else:
        print("Specify --model, --dataset, or --all")
        return

    print(f"{'Model':<12} {'Dataset':<22} {'WER':>8} {'MAR (Qwen P2)':>15} {'N valid':>8}")
    print("-" * 70)
    for model, dataset in targets:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        mar, n_valid, n_total, corpus_wer = process_file(path)
        mar_str = f"{mar*100:.2f}%" if mar is not None else "no verdicts"
        wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
        print(f"  {model:<10} {dataset:<22} {wer_str:>8} {mar_str:>15} {n_valid:>8}/{n_total}")

if __name__ == "__main__":
    main()