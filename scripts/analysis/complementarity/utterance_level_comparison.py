"""
utterance_level_comparison.py

When Qwen is wrong (sample_WER > 0), how often does each alternative model
have lower sample_WER than Qwen? Uses sample_WER from benchmark JSONs directly
to ensure consistent normalisation across all scripts.

Also computes: how often is AT LEAST ONE alternative better than Qwen —
this is the ceiling for any correction-based pipeline.

Usage:
    python scripts/utterance_level_comparison.py --dataset commonvoice
    python scripts/utterance_level_comparison.py --dataset all
    python scripts/utterance_level_comparison.py --dataset commonvoice --examples
"""

import json
import os
import argparse
from jiwer import wer

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

ALTERNATIVE_MODELS = ["whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

def analyse_dataset(dataset: str) -> dict:
    print(f"\n  Loading {dataset}...")

    model_samples = {}
    corpus_wers   = {}
    for model in ["qwen"] + ALTERNATIVE_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            data = json.load(f)
        model_samples[model] = data["samples"]
        corpus_wers[model]   = data.get("corpus_wer")

    n = len(model_samples["qwen"])

    total_valid  = 0
    qwen_correct = 0
    qwen_wrong   = 0

    alt_better_when_qwen_wrong     = {m: 0 for m in ALTERNATIVE_MODELS}
    any_alt_better_when_qwen_wrong = 0
    rescue_examples                = {m: [] for m in ALTERNATIVE_MODELS}

    for i in range(n):
        ref = model_samples["qwen"][i]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        q_wer = model_samples["qwen"][i].get("sample_WER")
        if q_wer is None:
            continue

        total_valid += 1

        alt_wers = {}
        for model in ALTERNATIVE_MODELS:
            a_wer = model_samples[model][i].get("sample_WER")
            if a_wer is None:
                continue
            alt_wers[model] = a_wer

        if q_wer == 0:
            qwen_correct += 1
            continue

        qwen_wrong += 1
        any_better = False

        for model in ALTERNATIVE_MODELS:
            if model not in alt_wers:
                continue
            if alt_wers[model] < q_wer:
                alt_better_when_qwen_wrong[model] += 1
                any_better = True

                if len(rescue_examples[model]) < 3:
                    rescue_examples[model].append({
                        "ref":      ref[:100],
                        "qwen":     model_samples["qwen"][i]["hyp"][:100],
                        "alt":      model_samples[model][i]["hyp"][:100],
                        "qwen_wer": round(q_wer, 3),
                        "alt_wer":  round(alt_wers[model], 3),
                    })

        if any_better:
            any_alt_better_when_qwen_wrong += 1

    return {
        "dataset":                        dataset,
        "total_valid":                    total_valid,
        "qwen_correct":                   qwen_correct,
        "qwen_wrong":                     qwen_wrong,
        "alt_better_when_qwen_wrong":     alt_better_when_qwen_wrong,
        "any_alt_better_when_qwen_wrong": any_alt_better_when_qwen_wrong,
        "corpus_wers":                    corpus_wers,
        "rescue_examples":                rescue_examples,
    }

def print_results(r: dict, show_examples: bool = False):
    ds    = r["dataset"]
    qw    = r["qwen_wrong"]
    total = r["total_valid"]

    print(f"\n{'='*65}")
    print(f"Dataset: {ds}  ({total} clips)")
    print(f"{'='*65}")

    print(f"\n── Corpus WER per model ──")
    for m, w in r["corpus_wers"].items():
        wstr = f"{w*100:.2f}%" if w is not None else "—"
        print(f"  {m:<12} {wstr}")

    print(f"\n── Qwen correctness ──")
    print(f"  Qwen correct (WER=0): {r['qwen_correct']:>5} ({r['qwen_correct']/total*100:.1f}%)")
    print(f"  Qwen wrong  (WER>0):  {qw:>5} ({qw/total*100:.1f}%)")

    if qw == 0:
        print("  Qwen is never wrong — no correction needed.")
        return

    print(f"\n── When Qwen is wrong ({qw} clips), how often is each alternative better? ──")
    for model in ALTERNATIVE_MODELS:
        n_better = r["alt_better_when_qwen_wrong"][model]
        pct      = n_better / qw * 100
        bar      = "█" * int(pct / 2)
        print(f"  {model:<12} {n_better:>5} / {qw} ({pct:>5.1f}%)  {bar}")

    any_n   = r["any_alt_better_when_qwen_wrong"]
    any_pct = any_n / qw * 100
    print(f"\n── At least ONE alternative better than Qwen when Qwen is wrong ──")
    print(f"  {any_n} / {qw} clips ({any_pct:.1f}%)")
    print(f"  → Ceiling for any correction-based pipeline")

    if show_examples:
        for model in ALTERNATIVE_MODELS:
            exs = r["rescue_examples"].get(model, [])
            if exs:
                print(f"\n  Examples where {model} beats Qwen:")
                for ex in exs:
                    print(f"    REF:   {ex['ref']}")
                    print(f"    QWEN:  {ex['qwen']}  (WER={ex['qwen_wer']})")
                    print(f"    {model.upper()}: {ex['alt']}  (WER={ex['alt_wer']})")
                    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="all",
                        choices=["commonvoice", "edacc", "english_dialects", "all"])
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_results = {}
    for dataset in datasets:
        r = analyse_dataset(dataset)
        print_results(r, show_examples=args.examples)
        all_results[dataset] = r

    if len(datasets) > 1:
        print(f"\n{'='*65}")
        print("CROSS-DATASET SUMMARY")
        print(f"{'='*65}")
        print(f"{'Dataset':<22} {'Qwen wrong%':>12} {'Whisper%':>10} {'Parakeet%':>11} {'wav2vec2%':>11} {'Ceiling%':>10}")
        for ds, r in all_results.items():
            qw   = r["qwen_wrong"]
            tot  = r["total_valid"]
            wh   = r["alt_better_when_qwen_wrong"]["whisper"]  / qw * 100 if qw else 0
            pa   = r["alt_better_when_qwen_wrong"]["parakeet"] / qw * 100 if qw else 0
            w2   = r["alt_better_when_qwen_wrong"]["wav2vec2"] / qw * 100 if qw else 0
            ceil = r["any_alt_better_when_qwen_wrong"]          / qw * 100 if qw else 0
            print(f"  {ds:<20} {qw/tot*100:>11.1f}% {wh:>9.1f}% {pa:>10.1f}% {w2:>10.1f}% {ceil:>9.1f}%")

    print()

if __name__ == "__main__":
    main()