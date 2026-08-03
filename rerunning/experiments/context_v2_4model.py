"""
rerunning/experiments/context_v2_4model.py

LAST-MINUTE EXPERIMENT - NOT part of the locked dissertation results.
Same rationale as context_v1_4model.py - adds wav2vec2 as a 4th voice
to test whether context_v2's gap to naive was about input count, not
the technique. auto_rules mechanism (build_rules_text) is unchanged -
that's a separate rule-generation system, unaffected by adding a 4th
transcript to the selector prompt itself.

Writes to writeup_results/experiments/context_v2_4model/ - SEPARATE
from writeup_results/ensembles/context_v2/.

Usage:
    python rerunning/experiments/context_v2_4model.py --dataset commonvoice --split dev --selector gemma4
"""

import json
import os
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.rules import build_rules_text
from src.splits import get_indices_for_split

OUTPUT_DIR = "writeup_results/experiments/context_v2_4model"
OLLAMA_HOST = "http://localhost:11434"
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript. You are given four transcripts of the same audio from different models.

TRANSCRIPT A (base - use this as your starting point, model: qwen):
{qwen}

TRANSCRIPT B (model: whisperx):
{whisperx}

TRANSCRIPT C (model: parakeet):
{parakeet}

TRANSCRIPT D (model: wav2vec2):
{wav2vec2}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE - applies to all words except where a specific rule below overrides it:
Only change a word if AT LEAST TWO of B, C, D disagree with A and agree with each other on the same alternative. If fewer than two of B, C, D agree on a different word, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates across all benchmark datasets):
{auto_rules}

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the four transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp, wav2vec2_hyp, auto_rules, num_predict, retries=2):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        qwen=qwen_hyp, whisperx=whisperx_hyp, parakeet=parakeet_hyp,
        wav2vec2=wav2vec2_hyp, auto_rules=auto_rules,
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (selector): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, auto_rules, max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]
    print(f"\n-- {dataset} | context_v2_4model (EXPERIMENT) selector={selector_key} split={split} --")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    wav2vec2_samples = get_indexed_samples("wav2vec2", dataset)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split: {split} ({len(indices)} samples)")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"context_v2_4model_{dataset}_{selector_key}_{split}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    for pos in range(start_from, len(indices)):
        idx = indices[pos]

        qwen_sample     = qwen_samples.get(idx)
        whisperx_sample = whisperx_samples.get(idx)
        parakeet_sample = parakeet_samples.get(idx)
        wav2vec2_sample = wav2vec2_samples.get(idx)

        if not all([qwen_sample, whisperx_sample, parakeet_sample, wav2vec2_sample]):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = qwen_sample["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "ignore_time_segment", "dataset_index": idx})
            continue

        if is_tag_only(ref):
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "tag_only_reference", "dataset_index": idx})
            continue

        qwen_hyp     = qwen_sample["hyp"]
        whisperx_hyp = whisperx_sample["hyp"]
        parakeet_hyp = parakeet_sample["hyp"]
        wav2vec2_hyp = wav2vec2_sample["hyp"]
        num_predict  = compute_num_predict([qwen_hyp, whisperx_hyp, parakeet_hyp, wav2vec2_hyp])

        best_hyp = ollama_select(client, selector_model, qwen_hyp, whisperx_hyp, parakeet_hyp,
                                  wav2vec2_hyp, auto_rules, num_predict)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref": ref, "hyp": best_hyp,
            "qwen_base": qwen_hyp, "whisperx_hyp": whisperx_hyp,
            "parakeet_hyp": parakeet_hyp, "wav2vec2_hyp": wav2vec2_hyp,
            "sample_WER": sample_wer_val, "severity": None,
            "auto_rules_used": auto_rules, "dataset_index": idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None

    output = {
        "selector": selector_key, "approach": "context_v2_4model",
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset, "split": split, "auto_rules_used": auto_rules,
        "subset_indices": indices, "corpus_wer": corpus_wer,
        "num_samples": len(valid), "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    print(f"\n  Now run severity: python rerunning/add_severity_to_existing_concurrent.py --files {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector", default="gemma4", choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()
