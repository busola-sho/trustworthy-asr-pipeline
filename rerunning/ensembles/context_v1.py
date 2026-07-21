"""
rerunning/ensembles/context_v1.py

PHASE 1 of 2: Context-aware profile-guided selector V1 — hand-written
error-pattern rules. Uses Qwen3-ASR as anchor, WhisperX and Parakeet as
supporting models. Selector calls only - no severity judging here.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

CHANGES from the original run_context_selector.py:
  - "whisper" replaced with "whisperx" throughout (standing decision)
  - Two-pass execution - selector calls only, severity judging separate
  - num_predict sized dynamically per sample; keep_alive="30m"; sleep removed
  - Added tag-only reference skip (e.g. "<OVERLAP>")
  - Added --split {dev,test,full} - defaults to "dev" for iteration
  - Writes to BOTH writeup_results/ensembles/context_v1/ (new) and
    results/combinations_v2judge/context/ (old, kept for continuity)

NOT CHANGED - NEEDS YOUR REVIEW: the rule content itself (the
"Whisper (B) is more reliable on named entities" line, and the other
hand-written error-pattern rules) is left exactly as written. You flagged
that WhisperX's relative reliability vs Qwen may vary by dataset - worth
checking per-dataset (not pooled) error profiles via src/rules.py before
deciding how to rewrite these rules. Edit SELECTOR_PROMPT below once
you've decided.

Usage:
    python rerunning/ensembles/context_v1.py --dataset commonvoice --split dev
    python rerunning/ensembles/context_v1.py --dataset edacc --split full --selector gemma2
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

NEW_OUTPUT_DIR = "writeup_results/ensembles/context_v1"
OLD_OUTPUT_DIR = "results/combinations_v2judge/context"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

# NOTE: rule content unchanged - review per the discussion above before editing.
SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point):
{qwen}

TRANSCRIPT B (WhisperX):
{whisperx}

TRANSCRIPT C (Parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except named entities:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

NAMED ENTITY RULE — applies to people's names, place names, organisations:
WhisperX (B) is more reliable on named entities. If B has a different named entity than A, consider switching — BUT only if C does not agree with A (case-insensitive). If C agrees with A on the named entity, keep A.

ADDITIONAL KNOWN ERROR PATTERNS:
- PROFANITY AND INFORMAL EXPRESSIONS: A sometimes self-censors mild profanity (e.g. "shit-scared"→"scared", "bloody"→"body", "sweet F all"→"sweetie fall") — if B and C have the original expression, restore it.
- NEGATIONS: If A drops or changes a negation and B and C preserve it — this is critical, switch.
- NUMBERS: If A has a different number than B and C agree on — switch.
- SCOTTISH DIALECT WORDS: If B or C preserve a Scottish dialect word that A has normalised (e.g. "wee", "wisnae", "dinnae", "cannae", "braw", "aboot", "carry-out", "noo") and both agree — preserve the dialect word.
- PRONOUNS: If A changes a pronoun (I/you/we/she/they) and B and C agree on the original — switch.

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(qwen=qwen_hyp, whisperx=whisperx_hyp, parakeet=parakeet_hyp)
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096, "num_predict": num_predict},
                keep_alive="30m",
            )
            return response.message.content.strip()
        except Exception as e:
            if attempt == retries:
                print(f"  ERROR (select): {e}")
                return None
    return None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V1 (PHASE 1: selector only) selector={selector_key} ──")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"context_{dataset}_{selector_key}_{split}.json"
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

        qwen_sample     = qwen_samples.get(idx)
        whisperx_sample = whisperx_samples.get(idx)
        parakeet_sample = parakeet_samples.get(idx)

        if not all([qwen_sample, whisperx_sample, parakeet_sample]):
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
        num_predict  = compute_num_predict([qwen_hyp, whisperx_hyp, parakeet_hyp])

        best_hyp = ollama_select(client, selector_model, qwen_hyp, whisperx_hyp, parakeet_hyp, num_predict)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":            ref,
            "hyp":            best_hyp,
            "qwen_base":      qwen_hyp,
            "whisperx_hyp":   whisperx_hyp,
            "parakeet_hyp":   parakeet_hyp,
            "sample_WER":     sample_wer_val,
            "severity":       None,
            "dataset_index":  idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "selector":       selector_key,
        "approach":       "context_v1",
        "phase":          "selector_only - severity not yet judged",
        "dataset":        dataset,
        "full_dataset":   full,
        "subset_indices": indices,
        "corpus_wer":     corpus_wer,
        "num_samples":    len(valid),
        "samples":        results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2:")
    print(f"  python rerunning/add_severity_to_existing.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--full",        action="store_true")
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