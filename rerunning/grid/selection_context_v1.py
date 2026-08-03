"""
rerunning/grid/selection_context_v1.py

Strategy: SELECTION - choose one existing candidate transcript unchanged.
Context condition: V1 - hand-written error-pattern rules, reframed from
context_v1.py's edit-trigger phrasing ("switch to B's version if...")
into relative judging criteria, since Selection cannot edit words -
only choose one whole candidate. Rule CONTENT is unchanged from
context_v1.py; only the framing (edit -> weigh evidence) differs,
which is a necessary adaptation for this strategy, not a change to
what V1 means as an experimental factor.

IMPORTANT - kept anchor-free, unlike context_v1.py: no candidate is
treated as a "default" or "base" here (context_v1.py explicitly starts
from Qwen and corrects it - that's Anchored Correction's defining
trait). Every candidate is named only by its source model, disclosed
per sample after rotation - this keeps Selection genuinely anchor-free
while still carrying V1's actual rule content as the context factor.

Uses all 4 ASR models (qwen, whisperx, parakeet, wav2vec2). Output is
LABEL-ONLY (A/B/C/D) - see selection_naive.py's docstring for why.

Writes to writeup_results/grid/selection_context_v1/.

Usage:
    python rerunning/grid/selection_context_v1.py --dataset commonvoice --split dev --selector gemma4
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

OUTPUT_DIR = "writeup_results/grid/selection_context_v1"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]
LABELS = ["A", "B", "C", "D"]

SELECTOR_PROMPT = """You are given four candidate ASR transcripts produced from the same spoken audio, each from a different ASR model.

Your task is to select the SINGLE candidate that is most likely to match what was spoken.

MODEL-SPECIFIC RELIABILITY NOTES:
- WhisperX is generally more reliable on named entities (people's names, place names, organisations). If candidates disagree on a named entity, give extra weight to whichever candidate matches WhisperX's version - unless a third candidate corroborates a different version instead, in which case treat it as genuinely disputed.
- Qwen sometimes self-censors mild profanity (e.g. "shit-scared"->"scared", "bloody"->"body", "sweet F all"->"sweetie fall"). If another candidate preserves the original expression and this is corroborated elsewhere, prefer that candidate.

GENERAL RULE - applies to all other disputed words:
Give more weight to a candidate whose disputed words are corroborated by at least two of the other three candidates. If no other candidate matches a given disputed word, do not treat that alone as disqualifying.

ADDITIONAL KNOWN ERROR PATTERNS - use these to weigh candidates:
- NEGATIONS: prefer whichever version most candidates support - dropped or added negation is a critical, meaning-altering difference.
- NUMBERS: prefer whichever number is corroborated by other candidates.
- SCOTTISH DIALECT WORDS: if a candidate preserves a Scottish dialect word (e.g. "wee", "wisnae", "dinnae", "cannae", "braw", "aboot", "carry-out", "noo") that another candidate has normalised, and this is corroborated by a third candidate, prefer the one preserving it.
- PRONOUNS: prefer whichever pronoun (I/you/we/she/they) most candidates support.

You must select one complete candidate transcript exactly as provided.
Return exactly one label: A, B, C, or D. No explanation, no other text.

CANDIDATE A (model: {model_a}):
{a}

CANDIDATE B (model: {model_b}):
{b}

CANDIDATE C (model: {model_c}):
{c}

CANDIDATE D (model: {model_d}):
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
    return 10


def parse_label(raw: str):
    stripped = raw.strip().upper()
    if stripped in LABELS:
        return stripped, True
    match = re.search(r"\b([ABCD])\b", stripped)
    if match:
        return match.group(1), False
    return None, False


def ollama_select(client, model_name, hyp_by_label, model_names_by_label, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(
        a=hyp_by_label["A"], model_a=model_names_by_label["A"],
        b=hyp_by_label["B"], model_b=model_names_by_label["B"],
        c=hyp_by_label["C"], model_c=model_names_by_label["C"],
        d=hyp_by_label["D"], model_d=model_names_by_label["D"],
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
    print(f"\n-- {dataset} | selection_context_v1 selector={selector_key} split={split} --")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"selection_context_v1_{dataset}_{selector_key}_{split}.json"
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
        model_names_by_label = {label: model_order[i] for i, label in enumerate(LABELS)}
        label_to_model = model_names_by_label

        raw_response = ollama_select(client, selector_model, hyp_by_label, model_names_by_label, num_predict)

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
        "approach": "selection_context_v1",
        "strategy": "selection",
        "context_condition": "v1",
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
