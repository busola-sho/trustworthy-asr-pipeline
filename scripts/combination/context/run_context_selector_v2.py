"""
run_context_selector_v2.py

Context-aware profile-guided selector V2 — auto-generated error-pattern rules
from the automated eval suite. Uses Qwen3-ASR as anchor, Whisper and Parakeet
as supporting models. Restricted to the 150-sample subset (seed=42).
Output saved to results/combinations_v2judge/context_v2/.

Usage:
    python scripts/combination/run_context_selector_v2.py --dataset commonvoice
    python scripts/combination/run_context_selector_v2.py --dataset edacc --gap 0.5
"""

import json
import os
import argparse
import time
import sys
from jiwer import wer
from ollama import Client

from src.judge import normalise, ollama_mar
from src.selector import (
    get_subset_indices, CANONICAL_FILES, OLLAMA_MODELS,
    DATASET_SIZES, check_selector_available,
)

from src.rules import build_rules_text

OUTPUT_DIR  = "results/combinations_v2judge/context_v2"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point, model: qwen):
{qwen}

TRANSCRIPT B (model: whisper):
{whisper}

TRANSCRIPT C (model: parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except where a specific rule below overrides it:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates across all benchmark datasets):
{auto_rules}

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def ollama_select(client, model_name, qwen_hyp, whisper_hyp, parakeet_hyp, auto_rules):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        qwen=qwen_hyp, whisper=whisper_hyp,
        parakeet=parakeet_hyp, auto_rules=auto_rules,
    )
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (selector): {e}")
        return None


def run_dataset(dataset, selector_key, client, auto_rules, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V2 selector={selector_key} ──")

    qwen_samples     = json.load(open(CANONICAL_FILES[("qwen",     dataset)]))["samples"]
    whisper_samples  = json.load(open(CANONICAL_FILES[("whisper",  dataset)]))["samples"]
    parakeet_samples = json.load(open(CANONICAL_FILES[("parakeet", dataset)]))["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"context_v2_{dataset}_{selector_key}_sub150.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    for pos in range(start_from, len(indices)):
        idx          = indices[pos]
        ref          = qwen_samples[idx]["ref"]
        qwen_hyp     = qwen_samples[idx]["hyp"]
        whisper_hyp  = whisper_samples[idx]["hyp"]
        parakeet_hyp = parakeet_samples[idx]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        best_hyp = ollama_select(client, selector_model, qwen_hyp,
                                 whisper_hyp, parakeet_hyp, auto_rules)
        time.sleep(0.05)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "qwen_base":       qwen_hyp,
            "whisper_hyp":     whisper_hyp,
            "parakeet_hyp":    parakeet_hyp,
            "sample_WER":      sample_wer_val,
            "qwen_verdict_p2": verdict,
            "auto_rules_used": auto_rules,
            "dataset_index":   idx,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{len(indices)} done")

    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "approach":                "context_v2",
        "dataset":                 dataset,
        "auto_rules_used":         auto_rules,
        "subset_indices":          indices,
        "corpus_wer":              corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    mar_str = f"{mar*100:.2f}%"        if mar        is not None else "—"
    print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    return {"dataset": dataset, "selector": selector_key,
            "wer": corpus_wer, "mar": mar, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--gap",         type=float, default=1.0,
                        help="Min gap (pp) for eval-suite rule generation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector}")
        print(f"Auto-generated rules:\n{auto_rules}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Using auto-generated rules:\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()