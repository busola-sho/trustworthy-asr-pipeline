"""
rerunning/grid/unanchored_fusion_context_v2.py

Strategy: UNANCHORED FUSION - construct a new transcript from all
candidates without treating one as the default (matches naive.py's
mechanics exactly). Context condition: V2 - auto-generated error-pattern
rules (build_rules_text), kept VERBATIM - identical text to what
context_v2.py and selection_context_v2.py use, not reframed or edited.

KNOWN LIMITATION - documented, not fixed (same as selection_context_v2.py):
build_rules_text()'s output contains rules phrased as a per-model
preference derived from measured error rates (e.g. "the qwen3asr
transcript is empirically more reliable on negations... lean toward
trusting it"). This is a mild content-level bias toward qwen3asr for
specific error categories, baked into V2 itself - not something
introduced by pairing V2 with Unanchored Fusion specifically. Kept
verbatim so V2 means the same thing across all three strategies in
the grid, rather than editing it per-strategy and contaminating the
strategy comparison instead.

Uses all 4 ASR models, candidate order rotated per sample (seeded by
dataset_index) - same as naive.py/unanchored_fusion_context_v1.py.

Writes to writeup_results/grid/unanchored_fusion_context_v2/.

Usage:
    python rerunning/grid/unanchored_fusion_context_v2.py --dataset commonvoice --split dev --selector gemma4
"""

import json
import os
import random
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split

OUTPUT_DIR = "writeup_results/clean_grid_guidance_rerun/unanchored_fusion_context_v2"
DEFAULT_RULES_FILE = "results/eval_suite/selector_rules_dev.txt"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are given four ASR transcripts of the same spoken audio, each from a different ASR model.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

GENERAL RULE - applies to all words except where a specific rule below overrides it:
Only change a word if at least two of the other three transcripts disagree with it and agree with each other on the same alternative.

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates on the development data):
{auto_rules}

Return only the final transcript, nothing else."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, model_order, hyp_by_model, auto_rules, num_predict, retries=2):
    hyp_block = "\n".join(
        f"Transcript {i+1} (model: {m}): {hyp_by_model[m]}" for i, m in enumerate(model_order)
    )
    system_prompt = SELECTOR_PROMPT_TEMPLATE.format(auto_rules=auto_rules)
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": hyp_block},
                ],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                keep_alive="30m",
                think=False,
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (select): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, auto_rules, max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]
    print(f"\n-- {dataset} | unanchored_fusion_context_v2 selector={selector_key} split={split} --")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"unanchored_fusion_context_v2_{dataset}_{selector_key}_{split}.json"
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
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = samples_by_model["qwen"]["ref"]

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

        hyp_by_model = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        best_hyp = ollama_select(client, selector_model, model_order, hyp_by_model, auto_rules, num_predict)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref": ref,
            "hyp": best_hyp,
            "source_hyps": hyp_by_model,
            "model_order": model_order,
            "sample_WER": sample_wer_val,
            "severity": None,
            "auto_rules_used": auto_rules,
            "dataset_index": idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None

    output = {
        "selector": selector_key,
        "approach": "unanchored_fusion_context_v2",
        "strategy": "unanchored_fusion",
        "context_condition": "v2",
        "asr_models": ASR_MODELS,
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset,
        "split": split,
        "auto_rules_used": auto_rules,
        "subset_indices": indices,
        "corpus_wer": corpus_wer,
        "num_samples": len(valid),
        "samples": results,
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
    parser.add_argument("--rules-file", default=DEFAULT_RULES_FILE,
                        help="Frozen dev-derived rules file")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.rules_file):
        raise SystemExit(f"Frozen rules file not found: {args.rules_file}")
    with open(args.rules_file, encoding="utf-8") as file:
        auto_rules = file.read().strip()
    if not auto_rules:
        raise SystemExit(f"Frozen rules file is empty: {args.rules_file}")

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Using auto-generated rules (verbatim, kept identical to context_v2.py):\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()
