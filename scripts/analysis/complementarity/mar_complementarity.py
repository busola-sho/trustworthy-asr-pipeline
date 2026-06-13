"""
mar_complementarity.py

When Qwen produces a meaning-altering error (qwen_verdict_p2 = true),
how often does each alternative model avoid it (qwen_verdict_p2 = false)?

This is semantic complementarity — the domain-relevant version of the
WER complementarity analysis.

Usage:
    python scripts/mar_complementarity.py --dataset commonvoice
    python scripts/mar_complementarity.py --dataset all
    python scripts/mar_complementarity.py --dataset commonvoice --examples
"""

import json
import os
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

ALTERNATIVE_MODELS = ["whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]

def analyse_dataset(dataset: str) -> dict:
    print(f"\n  Loading {dataset}...")

    model_samples = {}
    mar_rates     = {}
    for model in ["qwen"] + ALTERNATIVE_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
        with open(path) as f:
            data = json.load(f)
        model_samples[model] = data["samples"]
        # compute MAR from qwen_verdict_p2 per sample
        verdicts = [s.get("qwen_verdict_p2") for s in data["samples"]
                    if s.get("qwen_verdict_p2") is not None]
        mar_rates[model] = sum(verdicts) / len(verdicts) if verdicts else None

    n = len(model_samples["qwen"])

    total_valid          = 0
    qwen_ma              = 0  # qwen has meaning-altering error
    qwen_no_ma           = 0  # qwen has no meaning-altering error

    # when qwen has MA error: how often does each alternative avoid it?
    alt_avoids_when_qwen_ma = {m: 0 for m in ALTERNATIVE_MODELS}
    any_alt_avoids_when_qwen_ma = 0
    venn = {"whisper_only": 0, "parakeet_only": 0, "both": 0, "neither": 0}
    rescue_examples = {m: [] for m in ALTERNATIVE_MODELS}

    for i in range(n):
        ref       = model_samples["qwen"][i]["ref"]
        q_verdict = model_samples["qwen"][i].get("qwen_verdict_p2")

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue
        if q_verdict is None:
            continue

        total_valid += 1

        if not q_verdict:
            qwen_no_ma += 1
            continue

        # qwen has meaning-altering error
        qwen_ma += 1
        # track who fixes it
        whisper_fixes  = model_samples["whisper"][i].get("qwen_verdict_p2") == False
        parakeet_fixes = model_samples["parakeet"][i].get("qwen_verdict_p2") == False

        if whisper_fixes and parakeet_fixes:
            venn["both"] += 1
        elif whisper_fixes:
            venn["whisper_only"] += 1
        elif parakeet_fixes:
            venn["parakeet_only"] += 1
        else:
            venn["neither"] += 1

        any_avoids = whisper_fixes or parakeet_fixes

        if whisper_fixes:
            alt_avoids_when_qwen_ma["whisper"] += 1
        if parakeet_fixes:
            alt_avoids_when_qwen_ma["parakeet"] += 1

        wav2vec2_fixes = model_samples["wav2vec2"][i].get("qwen_verdict_p2") == False
        if wav2vec2_fixes:
            alt_avoids_when_qwen_ma["wav2vec2"] += 1

        if any_avoids:
            any_alt_avoids_when_qwen_ma += 1

    return {
        "dataset":                      dataset,
        "total_valid":                  total_valid,
        "qwen_ma":                      qwen_ma,
        "qwen_no_ma":                   qwen_no_ma,
        "alt_avoids_when_qwen_ma":      alt_avoids_when_qwen_ma,
        "any_alt_avoids_when_qwen_ma":  any_alt_avoids_when_qwen_ma,
        "venn":                         venn,
        "mar_rates":                    mar_rates,
        "rescue_examples":              rescue_examples,
    }

def print_results(r: dict, show_examples: bool = False):
    ds    = r["dataset"]
    qma   = r["qwen_ma"]
    total = r["total_valid"]

    print(f"\n{'='*65}")
    print(f"Dataset: {ds}  ({total} clips)")
    print(f"{'='*65}")

    print(f"\n── MAR per model ──")
    for m, rate in r["mar_rates"].items():
        rstr = f"{rate*100:.2f}%" if rate is not None else "—"
        print(f"  {m:<12} {rstr}")

    print(f"\n── Qwen MA errors ──")
    print(f"  Qwen meaning-altering:     {qma:>5} ({qma/total*100:.1f}%)")
    print(f"  Qwen no meaning-altering:  {r['qwen_no_ma']:>5} ({r['qwen_no_ma']/total*100:.1f}%)")

    if qma == 0:
        print("  Qwen never produces MA errors on this dataset.")
        return

    print(f"\n── When Qwen has MA error ({qma} clips), how often does each alternative avoid it? ──")
    for model in ALTERNATIVE_MODELS:
        n_avoids = r["alt_avoids_when_qwen_ma"][model]
        pct      = n_avoids / qma * 100
        bar      = "█" * int(pct / 2)
        print(f"  {model:<12} {n_avoids:>5} / {qma} ({pct:>5.1f}%)  {bar}")

    any_n   = r["any_alt_avoids_when_qwen_ma"]
    any_pct = any_n / qma * 100
    print(f"\n── At least ONE alternative avoids MA error when Qwen fails ──")
    print(f"  {any_n} / {qma} clips ({any_pct:.1f}%)")
    print(f"  → Ceiling for semantic correction-based pipeline")

    print(f"\n── Breakdown: who fixes Qwen's MA errors? ──")
    venn = r["venn"]
    for label, key in [
        ("Both Whisper and Parakeet fix it", "both"),
        ("Whisper only fixes it",            "whisper_only"),
        ("Parakeet only fixes it",           "parakeet_only"),
        ("Neither fixes it",                 "neither"),
    ]:
        n   = venn[key]
        pct = n / qma * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<38} {n:>5} ({pct:>5.1f}%)  {bar}")

    if show_examples:
        for model in ALTERNATIVE_MODELS:
            exs = r["rescue_examples"].get(model, [])
            if exs:
                print(f"\n  Examples where {model} avoids Qwen's MA error:")
                for ex in exs:
                    print(f"    REF:   {ex['ref']}")
                    print(f"    QWEN:  {ex['qwen']}")
                    print(f"    {model.upper()}: {ex['alt']}")
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
        print("CROSS-DATASET SUMMARY — MAR Complementarity")
        print(f"{'='*65}")
        print(f"{'Dataset':<22} {'Qwen MA%':>10} {'Whisper%':>10} {'Parakeet%':>11} {'wav2vec2%':>11} {'Ceiling%':>10}")
        for ds, r in all_results.items():
            qma  = r["qwen_ma"]
            tot  = r["total_valid"]
            wh   = r["alt_avoids_when_qwen_ma"]["whisper"]  / qma * 100 if qma else 0
            pa   = r["alt_avoids_when_qwen_ma"]["parakeet"] / qma * 100 if qma else 0
            w2   = r["alt_avoids_when_qwen_ma"]["wav2vec2"] / qma * 100 if qma else 0
            ceil = r["any_alt_avoids_when_qwen_ma"]          / qma * 100 if qma else 0
            print(f"  {ds:<20} {qma/tot*100:>9.1f}% {wh:>9.1f}% {pa:>10.1f}% {w2:>10.1f}% {ceil:>9.1f}%")

    print()

if __name__ == "__main__":
    main()