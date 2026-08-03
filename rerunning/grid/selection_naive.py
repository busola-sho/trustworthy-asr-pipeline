"""
rerunning/grid/selection_naive.py

Strategy: SELECTION - choose one existing candidate transcript unchanged.
Context condition: NAIVE - bare task instruction only, no extra guidance.

Part of the 9-cell strategy x context grid (see dissertation methods
plan). Uses all 4 ASR models (qwen, whisperx, parakeet, wav2vec2),
matching naive.py's input set - keeping ASR inputs identical across all
9 grid cells is required for the strategy comparison to be valid.

Output is LABEL-ONLY (A/B/C/D), not the full transcript text - this
guarantees selection purity by construction: the final hyp is always
byte-for-byte one of the 4 original candidates, never a model's
"helpful" rewording of it, which a full-text-copy instruction cannot
guarantee. Candidate order is rotated per sample (seeded by
dataset_index), same as naive.py, to avoid position bias - the returned
letter is mapped back through that sample's rotation to find which
model it actually refers to.

Writes to writeup_results/grid/selection_naive/ - separate from every
other existing ensemble output.

Usage:
    python rerunning/grid/selection_naive.py --dataset commonvoice --split dev --selector gemma4
"""

import json
import os
import re
import random
import argparse
from jiwer import wer
from ollama import Client

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split

OUTPUT_DIR = "writeup_results/grid/selection_naive"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]
LABELS = ["A", "B", "C", "D"]

SELECTOR_PROMPT = """You are given four candidate ASR transcripts produced from the same spoken audio.

Your task is to select the SINGLE candidate that is most likely to match what was spoken.

Evaluate the candidates using the evidence available across the transcripts. Consider:
- agreement between candidates;
- whether disputed words fit the surrounding linguistic context;
- whether a candidate contains likely ASR errors, such as substitutions, deletions, repetitions, or implausible wording;
- preservation of meaning-critical details, including negations, names, numbers, and key actions.

Return exactly one label: A, B, C, or D. No explanation, no other text.

CANDIDATE A:
{a}

CANDIDATE B:
{b}

CANDIDATE C:
{c}

CANDIDATE D:
{d}"""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict() -> int:
    # label-only output - fixed small cap, no need to scale with input length
    return 10


def parse_label(raw: str):
    """Returns the letter (A/B/C/D) if the response is compliant, else
    None. Compliance = the response, stripped, is exactly one of the
    four letters, OR the first standalone letter found via fallback
    regex - anything else is a genuine compliance failure, not guessed."""
    stripped = raw.strip().upper()
    if stripped in LABELS:
        return stripped, True
    match = re.search(r"\b([ABCD])\b", stripped)
    if match:
        return match.group(1), False   # compliant-ish, needed fallback parse
    return None, False


def ollama_select(client, model_name, hyp_by_label, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(
        a=hyp_by_label["A"], b=hyp_by_label["B"], c=hyp_by_label["C"], d=hyp_by_label["D"]
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
                print(f"  ERROR (select): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]
    print(f"\n-- {dataset} | selection_naive selector={selector_key} split={split} --")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"selection_naive_{dataset}_{selector_key}_{split}.json"
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

    num_predict = compute_num_predict()
    n_noncompliant = 0

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

        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        hyp_by_label = {label: samples_by_model[model_order[i]]["hyp"] for i, label in enumerate(LABELS)}
        label_to_model = {label: model_order[i] for i, label in enumerate(LABELS)}

        raw_response = ollama_select(client, selector_model, hyp_by_label, num_predict)

        if raw_response is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        chosen_label, was_clean = parse_label(raw_response)

        if chosen_label is None:
            n_noncompliant += 1
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": f"non-compliant response: '{raw_response[:80]}'",
                            "dataset_index": idx})
            continue

        if not was_clean:
            n_noncompliant += 1

        chosen_model = label_to_model[chosen_label]
        best_hyp = hyp_by_label[chosen_label]

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref": ref,
            "hyp": best_hyp,
            "chosen_label": chosen_label,
            "chosen_model": chosen_model,
            "model_order": model_order,
            "raw_response": raw_response,
            "response_was_clean": was_clean,
            "sample_WER": sample_wer_val,
            "severity": None,
            "dataset_index": idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None
    compliance = 1 - (n_noncompliant / len(valid)) if valid else None

    output = {
        "selector": selector_key,
        "approach": "selection_naive",
        "strategy": "selection",
        "context_condition": "naive",
        "asr_models": ASR_MODELS,
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset,
        "split": split,
        "subset_indices": indices,
        "corpus_wer": corpus_wer,
        "compliance": compliance,
        "num_samples": len(valid),
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    comp_str = f"{compliance*100:.1f}%" if compliance is not None else "-"
    print(f"\n  WER: {wer_str}  Compliance: {comp_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    print(f"\n  Now run severity: python rerunning/add_severity_to_existing_concurrent.py --files {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector", default="gemma4", choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()
