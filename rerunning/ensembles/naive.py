"""
rerunning/ensembles/naive.py

Naive combination — passes all 4 ASR model transcripts (WhisperX, Qwen,
Parakeet, wav2vec2) to a selector LLM (word-level combiner), scored with
the locked severity judge (Phi-4 + direct prompt).

CHANGES from the original run_naive_combination_v3.py:
  - "whisper" (plain Whisper) replaced with "whisperx" throughout, per the
    standing decision that WhisperX replaces Whisper everywhere going forward
  - Judge swapped from binary qwen2.5:7b MAR to the locked severity judge
    (Phi-4 + direct prompt, QWK=0.783)
  - Added tag-only reference skip (e.g. "<OVERLAP>"), matching the fix
    already applied to the individual-model benchmark scripts - the
    original script only skipped IGNORE_TIME_SEGMENT_IN_SCORING
  - Added --full mode for full-dataset runs (not the 150-sample subset)
  - Writes to BOTH writeup_results/ensembles/naive/ (new) and
    results/combinations_v2judge/ (old, kept for continuity)

IMPORTANT - NEEDS YOUR INPUT: this script relies on CANONICAL_FILES in
src/selector.py to resolve each model's transcript file per dataset. That
mapping currently points at "whisper" (not "whisperx") and likely still
points at subset files, not your new full-dataset benchmark outputs
(e.g. the timestamped whisperx_*.json files in writeup_results/benchmarks/main/).
Paste src/selector.py so this can be updated properly - every other
ensemble script will depend on the same fix, so getting it right once here
matters for all of them.

Usage:
    python rerunning/ensembles/naive.py --dataset commonvoice --full
    python rerunning/ensembles/naive.py --dataset edacc --full --selector gemma2
    python rerunning/ensembles/naive.py --dry-run
"""

import json
import os
import re
import argparse
import time
from jiwer import wer
from ollama import Client
from dotenv import load_dotenv

from src.judge import normalise, is_tag_only
from src.selector import (
    get_subset_indices, find_canonical_file, OLLAMA_MODELS,
    DATASET_SIZES, check_selector_available,
)
from severity_judge_prompts import DIRECT_SEVERITY_PROMPT

load_dotenv()

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive"
OLD_OUTPUT_DIR = "results/combinations_v2judge"
OLLAMA_HOST    = "http://localhost:11434"
JUDGE_MODEL    = "phi4:14b"   # locked severity judge (Phi-4 + direct, QWK=0.783)
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]   # whisper -> whisperx
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps

Return only the final transcript, nothing else."""


def parse_severity(text: str):
    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def ollama_severity(client: Client, ref: str, hyp: str, retries: int = 2):
    prompt = DIRECT_SEVERITY_PROMPT.format(reference=ref, hypothesis=hyp)
    text = None
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
                think=False,
            )
            text = response.message.content
            severity = parse_severity(text)
            if severity is not None:
                return severity, text
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (severity judge): {e}")
                return None, None
            time.sleep(1.0)
    return None, text


def ollama_select(client, model_name, hyps):
    hyp_block = "\n".join([f"Transcript {i+1}: {h}" for i, h in enumerate(hyps)])
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SELECTOR_PROMPT},
                {"role": "user",   "content": hyp_block},
            ],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (select): {e}")
        return None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, full=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive selector={selector_key} | judge={JUDGE_MODEL} direct ──")

    model_samples = {}
    for m in ASR_MODELS:
        path = CANONICAL_FILES[(m, dataset)]
        with open(path) as f:
            model_samples[m] = json.load(f)["samples"]

    if full:
        indices = list(range(DATASET_SIZES[dataset]))
        print(f"  FULL dataset mode: {len(indices)} samples")
    else:
        indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    suffix = "full" if full else "sub150"
    filename = f"naive_{dataset}_{selector_key}sel_{suffix}.json"
    new_output_path = os.path.join(NEW_OUTPUT_DIR, filename)
    old_output_path = os.path.join(OLD_OUTPUT_DIR, filename)

    if os.path.exists(old_output_path) and not rerun:
        with open(old_output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(new_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(old_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    for pos in range(start_from, len(indices)):
        idx = indices[pos]
        ref = model_samples["qwen"][idx]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "ignore_time_segment", "dataset_index": idx})
            continue

        # skip refs that are entirely bracketed annotation tags (e.g. "<OVERLAP>")
        if is_tag_only(ref):
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "tag_only_reference", "dataset_index": idx})
            continue

        hyps     = [model_samples[m][idx]["hyp"] for m in ASR_MODELS]
        best_hyp = ollama_select(client, selector_model, hyps)
        time.sleep(0.1)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        severity, raw_judge_response = ollama_severity(client, ref, best_hyp)

        results.append({
            "ref":                ref,
            "hyp":                best_hyp,
            "source_hyps":        {m: model_samples[m][idx]["hyp"] for m in ASR_MODELS},
            "severity":           severity,
            "judge_raw_response": raw_judge_response,
            "sample_WER":         sample_wer_val,
            "dataset_index":      idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    severities = [r["severity"] for r in valid if r.get("severity") is not None]
    mean_severity = sum(severities) / len(severities) if severities else None
    severity_distribution = {str(i): severities.count(i) for i in range(5)}

    output = {
        "selector":                selector_key,
        "approach":                "naive",
        "asr_models":              ASR_MODELS,
        "judge":                   f"{JUDGE_MODEL} (direct prompt)",
        "dataset":                 dataset,
        "full_dataset":            full,
        "subset_indices":          indices,
        "corpus_wer":              corpus_wer,
        "mean_severity":           round(mean_severity, 3) if mean_severity is not None else None,
        "severity_distribution":   severity_distribution,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    sev_str = f"{mean_severity:.3f}" if mean_severity is not None else "—"
    print(f"  WER: {wer_str}  Mean severity: {sev_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    return {"dataset": dataset, "selector": selector_key,
            "wer": corpus_wer, "mean_severity": mean_severity, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full",        action="store_true",
                        help="Run on the FULL dataset instead of the 150-sample subset")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} full={args.full}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, full=args.full)


if __name__ == "__main__":
    main()