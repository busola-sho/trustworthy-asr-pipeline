"""
inspect_errors.py

Qualitative error analysis on combination approach results.
For each remaining meaning-altering error, shows ref, all 4 model outputs,
and the combination hypothesis side by side — so you can see whether the
correct transcript was available in any model, or whether all models failed.

Optionally tag errors by category for frequency analysis.
Writes full output to a file for dissertation reference.

Usage:
    python scripts/analysis/inspect_errors.py --dataset commonvoice --approach context_v2_confidence
    python scripts/analysis/inspect_errors.py --dataset commonvoice --approach context_v2_confidence --tag
    python scripts/analysis/inspect_errors.py --dataset edacc --approach context_v1 --n 20
"""

import json
import os
import argparse
from datetime import datetime
from jiwer import wer
from src.judge import normalise
from src.selector import CANONICAL_FILES

RESULTS_DIR = "results/combinations_v2judge"

SOURCE_PATHS = {
    ("commonvoice", "baseline"):              "baseline/baseline_commonvoice_qwen_sub150.json",
    ("commonvoice", "naive"):                 "naive_commonvoice_qwensel_p3_qwenjud_sub150.json",
    ("commonvoice", "context_v1"):            "context/context_commonvoice_qwen_sub150.json",
    ("commonvoice", "context_v2"):            "context_v2/context_v2_commonvoice_qwen_sub150.json",
    ("commonvoice", "context_v1_confidence"): "context_v1_confidence/context_v1conf_commonvoice_qwen_t080_sub150.json",
    ("commonvoice", "context_v2_confidence"): "context_v2_confidence/context_v2conf_commonvoice_qwen_t080_sub150.json",
    ("commonvoice", "naive_confidence"):      "naive_confidence/naive_conf_commonvoice_qwen_t080_sub150.json",

    ("edacc", "baseline"):                    "baseline/baseline_edacc_qwen_sub150.json",
    ("edacc", "naive"):                       "naive_edacc_qwensel_p3_qwenjud_sub150.json",
    ("edacc", "context_v1"):                  "context/context_edacc_qwen_sub150.json",
    ("edacc", "context_v2"):                  "context_v2/context_v2_edacc_qwen_sub150.json",
    ("edacc", "context_v1_confidence"):       "context_v1_confidence/context_v1conf_edacc_qwen_t080_sub150.json",
    ("edacc", "context_v2_confidence"):       "context_v2_confidence/context_v2conf_edacc_qwen_t080_sub150.json",
    ("edacc", "naive_confidence"):            "naive_confidence/naive_conf_edacc_qwen_t080_sub150.json",

    ("english_dialects", "baseline"):         "baseline/baseline_english_dialects_qwen_sub150.json",
    ("english_dialects", "naive"):            "naive_english_dialects_qwensel_p3_qwenjud_sub150.json",
    ("english_dialects", "context_v1"):       "context/context_english_dialects_qwen_sub150.json",
    ("english_dialects", "context_v2"):       "context_v2/context_v2_english_dialects_qwen_sub150.json",
}

CATEGORIES = [
    "negation",
    "dialect_word",
    "named_entity",
    "hallucination",
    "number_date",
    "pronoun",
    "other",
]


def load_errors(path):
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])
    return [
        s for s in samples
        if s.get("qwen_verdict_p2") is True
        and not s.get("skipped")
        and not s.get("error")
        and s.get("ref") and s.get("hyp")
    ]


def load_all_model_samples(dataset):
    all_models = {}
    for model in ["qwen", "whisper", "parakeet", "wav2vec2"]:
        key = (model, dataset)
        if key in CANONICAL_FILES:
            try:
                with open(CANONICAL_FILES[key]) as f:
                    all_models[model] = json.load(f)["samples"]
            except FileNotFoundError:
                pass
    return all_models


def format_sample(i, total, sample, model_samples, dataset):
    lines = []
    ref = sample["ref"]
    hyp = sample["hyp"]
    idx = sample.get("dataset_index")
    sample_wer = sample.get("sample_WER", wer(normalise(ref), normalise(hyp)))

    lines.append(f"\n{'='*70}")
    lines.append(f"  Error {i+1}/{total}  |  WER: {sample_wer*100:.1f}%  |  dataset_index: {idx}")
    lines.append(f"{'='*70}")
    lines.append(f"REF:      {ref}")
    lines.append(f"")

    # show all 4 individual model outputs
    if idx is not None:
        for model in ["qwen", "whisper", "parakeet", "wav2vec2"]:
            if model in model_samples and idx < len(model_samples[model]):
                model_hyp = model_samples[model][idx].get("hyp", "—")
                model_wer = wer(normalise(ref), normalise(model_hyp)) if model_hyp else None
                wer_str = f"WER={model_wer*100:.1f}%" if model_wer is not None else ""
                lines.append(f"{model.upper():<10} {wer_str:<12} {str(model_hyp)}")

    lines.append(f"")
    lines.append(f"COMBO:    {hyp}")

    # note if selector changed anything from qwen base
    qwen_base = sample.get("qwen_base")
    if qwen_base and qwen_base != hyp:
        lines.append(f"[selector changed from qwen base]")
    else:
        lines.append(f"[selector kept qwen base unchanged]")

    return "\n".join(lines)


def tag_interactively(errors, model_samples, dataset, output_lines):
    tags = {}
    print("\nCategories:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  {i+1}. {cat}")
    print("  s. skip  |  q. quit tagging\n")

    for i, s in enumerate(errors):
        block = format_sample(i, len(errors), s, model_samples, dataset)
        print(block)
        output_lines.append(block)

        choice = input("Tag (1-7/s/q): ").strip().lower()
        if choice == "q":
            break
        elif choice == "s":
            continue
        elif choice.isdigit() and 1 <= int(choice) <= len(CATEGORIES):
            cat = CATEGORIES[int(choice) - 1]
            tags[i] = cat
            tag_line = f"  >> TAGGED: {cat}"
            print(tag_line)
            output_lines.append(tag_line)
        else:
            tags[i] = "other"

    return tags


def print_and_write_summary(errors, tags, output_lines):
    from collections import Counter
    counts = Counter(tags.values())
    total_tagged = len(tags)
    lines = [
        f"\n{'='*40}",
        f"ERROR CATEGORY BREAKDOWN ({total_tagged}/{len(errors)} tagged)",
        f"{'='*40}",
    ]
    for cat, count in counts.most_common():
        pct = count / total_tagged * 100
        lines.append(f"  {cat:<20} {count:>3}  ({pct:.1f}%)")
    summary = "\n".join(lines)
    print(summary)
    output_lines.append(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="commonvoice",
                        choices=["commonvoice", "edacc", "english_dialects"])
    parser.add_argument("--approach",  default="context_v2_confidence")
    parser.add_argument("--threshold", default=None,
                        help="Override threshold suffix e.g. 0.5 → t050")
    parser.add_argument("--n",         type=int, default=None,
                        help="Max errors to show")
    parser.add_argument("--sort",      default="wer",
                        choices=["wer", "index"])
    parser.add_argument("--tag",       action="store_true",
                        help="Interactively tag errors by category")
    args = parser.parse_args()

    key = (args.dataset, args.approach)
    if key not in SOURCE_PATHS:
        print(f"No path configured for {args.dataset}/{args.approach}")
        return

    rel_path = SOURCE_PATHS[key]
    if args.threshold:
        thresh_str = f"t{float(args.threshold):.2f}".replace(".", "")
        for old in ["t080", "t070", "t050"]:
            rel_path = rel_path.replace(old, thresh_str)

    path = os.path.join(RESULTS_DIR, rel_path)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    errors = load_errors(path)
    model_samples = load_all_model_samples(args.dataset)

    header = (
        f"Error analysis: {args.dataset} / {args.approach}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Total remaining errors: {len(errors)}\n"
    )
    print(header)
    output_lines = [header]

    if args.sort == "wer":
        errors.sort(key=lambda s: s.get("sample_WER", 0), reverse=True)
    if args.n:
        errors = errors[:args.n]

    if args.tag:
        tags = tag_interactively(errors, model_samples, args.dataset, output_lines)
        if tags:
            print_and_write_summary(errors, tags, output_lines)
    else:
        for i, s in enumerate(errors):
            block = format_sample(i, len(errors), s, model_samples, args.dataset)
            print(block)
            output_lines.append(block)
        print(f"\nShowed {len(errors)} errors. Re-run with --tag to categorise.")

    # write to file
    os.makedirs("results/error_analysis", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/error_analysis/{args.dataset}_{args.approach}_{ts}.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()