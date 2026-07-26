"""
rerunning/ensembles/whole_transcript_selection.py

BASELINE 4: whole-transcript LLM selection. The LLM must pick exactly
ONE of the four ASR transcripts, completely UNCHANGED - no editing, no
word-swapping, no combining across transcripts allowed at all. This is
a stricter version of naive.py (which permits light word/phrase
swapping between transcripts).

WHY THIS BASELINE MATTERS: it isolates whether naive's improvement over
the individual ASR models comes from genuine cross-transcript fusion
(combining the best parts of multiple transcripts), or whether the LLM
is essentially just good at picking the best single transcript and the
"fusion" aspect isn't actually doing much work. If this baseline scores
close to naive, that's evidence naive's editing capability isn't adding
much; if naive clearly beats this, that's evidence the editing/fusion
step is doing real work.

COMPLIANCE TRACKING: LLMs don't always follow "don't edit" instructions
perfectly even when explicitly told not to. Each sample's output is
checked against all 4 input transcripts (after normalisation) - if it
exactly matches one of them, "compliant": true and "matched_source"
records which model was picked. If it doesn't exactly match any of the
four (the LLM edited despite instructions), "compliant": false - this
compliance rate is itself a reportable result, not just a QA checkbox.

CONCURRENT: same Phase A (fast sequential prep) / Phase B (concurrent
Ollama calls via thread pool) pattern as your other ensemble scripts.
MAX_WORKERS reads from ENSEMBLE_MAX_WORKERS env var (default 8) or
--max-workers, same convention as the rest of the pipeline - lower this
on memory-limited machines (e.g. a 16GB Mac), higher on the HPC.

Usage:
    python rerunning/ensembles/whole_transcript_selection.py --dataset commonvoice --split dev
    python rerunning/ensembles/whole_transcript_selection.py --dataset edacc --split full --selector gemma4
    python rerunning/ensembles/whole_transcript_selection.py --dry-run
"""

import json
import os
import random
import argparse
from jiwer import wer
from ollama import Client
from dotenv import load_dotenv

from src.judge import normalise, is_tag_only
from src.selector import (
    find_canonical_file, OLLAMA_MODELS, check_selector_available, load_samples,
)
from src.splits import get_indices_for_split
from src.concurrent_ollama import run_concurrent

load_dotenv()

NEW_OUTPUT_DIR = "writeup_results/ensembles/whole_transcript_selection"
OLD_OUTPUT_DIR = "results/combinations_v2judge/whole_transcript_selection"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))
# tune via --max-workers or the ENSEMBLE_MAX_WORKERS env var (roughly match
# OLLAMA_NUM_PARALLEL on the server) - NOT a hardcoded constant, since your
# Mac (limited unified memory) and the HPC (dedicated GPU memory) need very
# different values, and this file is shared between both via git.

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to select the SINGLE most accurate transcript, completely UNCHANGED, from the four options.

You MUST NOT:
- Edit, correct, or modify any word in your chosen transcript
- Combine or merge parts of different transcripts together
- Add or remove any words
- Paraphrase, rewrite, or restructure anything

Return ONLY the exact text of your chosen transcript, copied verbatim character-for-character, nothing else. No explanation, no labels."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def get_rotated_order(models: list, seed_key: int) -> list:
    """Same deterministic per-sample shuffle as naive.py, for identical
    position-bias handling and reproducibility."""
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, model_order, hyp_by_model, num_predict, retries=2):
    hyp_block = "\n".join(
        f"Transcript {i+1}: {hyp_by_model[m]}" for i, m in enumerate(model_order)
    )
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
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


def check_compliance(chosen_hyp: str, hyp_by_model: dict):
    """
    Checks whether the LLM's output exactly matches one of the 4 input
    transcripts (after normalisation), as instructed. Returns
    (compliant: bool, matched_source: str|None).
    """
    normalised_chosen = normalise(chosen_hyp)
    for model, hyp in hyp_by_model.items():
        if normalise(hyp) == normalised_chosen:
            return True, model
    return False, None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False,
                 split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | whole_transcript_selection (PHASE 1: selector only, CONCURRENT) "
          f"selector={selector_key} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"wholetranscript_{dataset}_{selector_key}sel_{split}.json"
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

    # ── PHASE A: fast sequential prep - build work items, append skips directly ──
    remaining_indices = indices[start_from:]
    work_items = []   # each: (idx, ref, model_order, hyp_by_model, num_predict)
    skip_slots = {}

    for idx in remaining_indices:
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            skip_slots[idx] = {"ref": None, "hyp": None, "severity": None,
                               "sample_WER": None, "error": True,
                               "error_reason": "missing sample from one or more models",
                               "dataset_index": idx}
            continue

        ref = samples_by_model["qwen"]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "ignore_time_segment", "dataset_index": idx}
            continue

        if is_tag_only(ref):
            skip_slots[idx] = {"ref": ref, "hyp": None, "severity": None,
                               "sample_WER": None, "skipped": True,
                               "skip_reason": "tag_only_reference", "dataset_index": idx}
            continue

        hyp_by_model = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        work_items.append((idx, ref, model_order, hyp_by_model, num_predict))

    print(f"  {len(work_items)} samples queued for concurrent Ollama calls "
          f"({len(skip_slots)} skipped without needing a call)")

    # ── PHASE B: fire all Ollama calls concurrently ──
    def _worker(item):
        idx, ref, model_order, hyp_by_model, num_predict = item
        return ollama_select(client, selector_model, model_order, hyp_by_model, num_predict)

    best_hyps = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)

    # ── Reassemble in original index order, checking compliance ──
    call_results = {item[0]: (item, best_hyp) for item, best_hyp in zip(work_items, best_hyps)}

    n_compliant = 0
    n_noncompliant = 0

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, best_hyp = call_results[idx]
        _, ref, model_order, hyp_by_model, _ = item

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        compliant, matched_source = check_compliance(best_hyp, hyp_by_model)
        if compliant:
            n_compliant += 1
        else:
            n_noncompliant += 1

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":            ref,
            "hyp":            best_hyp,
            "source_hyps":    hyp_by_model,
            "model_order":    model_order,
            "sample_WER":     sample_wer_val,
            "severity":       None,
            "compliant":      compliant,
            "matched_source": matched_source,
            "dataset_index":  idx,
        })

        if len(results) % 10 == 0:
            save_progress()

    save_progress()

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    total_checked = n_compliant + n_noncompliant
    compliance_rate = (n_compliant / total_checked) if total_checked else None

    # per-model pick distribution, among compliant samples only (tells you
    # whether the selector is systematically favouring one ASR model)
    pick_distribution = {m: 0 for m in ASR_MODELS}
    for r in valid:
        if r.get("compliant") and r.get("matched_source"):
            pick_distribution[r["matched_source"]] += 1

    output = {
        "selector":          selector_key,
        "approach":          "whole_transcript_selection",
        "asr_models":        ASR_MODELS,
        "phase":             "selector_only - severity not yet judged",
        "dataset":           dataset,
        "split":             split,
        "subset_indices":    indices,
        "corpus_wer":        corpus_wer,
        "num_samples":       len(valid),
        "compliance_rate":   compliance_rate,
        "n_compliant":       n_compliant,
        "n_noncompliant":    n_noncompliant,
        "pick_distribution": pick_distribution,
        "samples":           results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    compliance_str = f"{compliance_rate*100:.1f}%" if compliance_rate is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Compliance (chose one transcript verbatim, no edits): {compliance_str} "
          f"({n_compliant}/{total_checked})")
    print(f"  Pick distribution among compliant samples: {pick_distribution}")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"],
                        help="'dev' for iteration (default, excludes calibration+test), "
                             "'full' only for a final confirmatory run, 'test' to check "
                             "the in-domain test split specifically")
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrent Ollama calls (default from ENSEMBLE_MAX_WORKERS "
                             "env var, or 8 if unset). Lower this on memory-limited machines "
                             "(e.g. a 16GB Mac) - try 1-2. Higher is fine on a dedicated GPU "
                             "node with real VRAM headroom.")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} split={args.split}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split,
                max_workers=args.max_workers)


if __name__ == "__main__":
    main()