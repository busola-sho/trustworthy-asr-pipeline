"""
rerunning/grid/anchored_correction_naive.py

Strategy: ANCHORED CORRECTION - start from Qwen and make targeted
corrections using the other candidates (matches context_v1.py/
context_v2.py's mechanics exactly - Qwen is always the fixed anchor,
never rotated, since that IS the definition of this strategy).
Context condition: NAIVE - bare task instruction only, no hand-written
or auto-generated rules at all.

Extended to 4 models (adds wav2vec2 as a 4th supporting transcript,
alongside WhisperX and Parakeet) for grid consistency with naive.py's
input set. GENERAL RULE is a majority vote across the 3 supporting
transcripts (at least 2 of B/C/D must agree on an alternative to
override Qwen) - the natural generalisation from context_v1/v2's
2-of-2 rule to a 3-voter setting, same generalisation used in the
context_v1_4model/context_v2_4model experiment scripts.

No rotation here - Qwen is always Transcript A / the anchor, by design.

Writes to writeup_results/grid/anchored_correction_naive/.

Usage:
    python rerunning/grid/anchored_correction_naive.py --dataset commonvoice --split dev --selector gemma4
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
from src.splits import get_indices_for_split

OUTPUT_DIR = "writeup_results/grid/anchored_correction_naive"
OLLAMA_HOST = "http://localhost:11434"
DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given four transcripts of the same audio from different models.

TRANSCRIPT A (base - use this as your starting point, model: qwen):
{qwen}

TRANSCRIPT B (model: whisperx):
{whisperx}

TRANSCRIPT C (model: parakeet):
{parakeet}

TRANSCRIPT D (model: wav2vec2):
{wav2vec2}

Your task: return Transcript A with targeted corrections where needed. Only change a word if at least two of B, C, D disagree with it and agree with each other on the same alternative.

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


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp, wav2vec2_hyp, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(qwen=qwen_hyp, whisperx=whisperx_hyp, parakeet=parakeet_hyp, wav2vec2=wav2vec2_hyp)
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
    print(f"\n-- {dataset} | anchored_correction_naive selector={selector_key} split={split} --")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    wav2vec2_samples = get_indexed_samples("wav2vec2", dataset)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"anchored_correction_naive_{dataset}_{selector_key}_{split}.json"
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

        best_hyp = ollama_select(client, selector_model, qwen_hyp, whisperx_hyp, parakeet_hyp, wav2vec2_hyp, num_predict)

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
            "dataset_index": idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error") and r.get("sample_WER") is not None]
    corpus_wer = wer([normalise(r["ref"]) for r in valid], [normalise(r["hyp"]) for r in valid]) if valid else None

    output = {
        "selector": selector_key,
        "approach": "anchored_correction_naive",
        "strategy": "anchored_correction",
        "context_condition": "naive",
        "asr_models": ["qwen", "whisperx", "parakeet", "wav2vec2"],
        "phase": "selector_only - severity not yet judged",
        "dataset": dataset,
        "split": split,
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
