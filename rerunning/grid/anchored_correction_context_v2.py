"""
rerunning/grid/anchored_correction_context_v2.py

PHASE 1 of 2: Context-aware profile-guided selector V2 — auto-generated
error-pattern rules from the automated eval suite. Uses Qwen3-ASR as
anchor, with WhisperX, Parakeet, and wav2vec2 as supporting models.
Selector calls only - no severity judging here.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing_concurrent.py --files <output file from this script>

FIX (this version): same bug as anchored_correction_context_v1.py -
filename pattern was still "context_v2_{...}", left over from the
original context_v2.py this was adapted from - only NEW_OUTPUT_DIR had
been updated to the new grid folder, not the filename itself. Filename
and "approach" field now both correctly say "anchored_correction_v2",
matching the rest of the grid and the folder name.

Usage:
    python rerunning/grid/anchored_correction_context_v2.py --dataset commonvoice --split dev
    python rerunning/grid/anchored_correction_context_v2.py --dataset edacc --split full --gap 0.5
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

NEW_OUTPUT_DIR = "writeup_results/grid/anchored_correction_v2"
OLD_OUTPUT_DIR = "results/combinations_v2judge/context_v2"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript. You are given four transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point, model: Qwen):
{qwen}

TRANSCRIPT B (model: WhisperX):
{whisperx}

TRANSCRIPT C (model: Parakeet):
{parakeet}

TRANSCRIPT D (model: wav2vec2):
{wav2vec2}

Your task is to return Transcript A with targeted corrections where needed.

GENERAL RULE — applies unless a model-specific rule below overrides it:
Only change a word or short phrase in Transcript A when at least two of Transcripts B, C, and D disagree with A and agree with each other on the same alternative. If fewer than two supporting transcripts agree on an alternative, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates on the development data):
{auto_rules}

Apply a model-specific rule only to the error category it describes. Otherwise, follow the general rule above.

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the four transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_num_predict(hyps: list) -> int:
    """Size num_predict dynamically per sample, based on the longest input
    transcript - avoids truncating long EdAcc/English Dialects monologues."""
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp,
                   wav2vec2_hyp, auto_rules, num_predict, retries=2):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        qwen=qwen_hyp, whisperx=whisperx_hyp,
        parakeet=parakeet_hyp, wav2vec2=wav2vec2_hyp,
        auto_rules=auto_rules,
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


def run_dataset(dataset, selector_key, client, auto_rules, max_samples=None,
                 rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | anchored correction V2 (PHASE 1: selector only) selector={selector_key} split={split} ──")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    wav2vec2_samples = get_indexed_samples("wav2vec2", dataset)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split: {split} ({len(indices)} samples)")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    # FIX: was "context_v2_{dataset}_{selector_key}_{split}.json" - now
    # matches the approach name and folder name consistently.
    filename = f"anchored_correction_v2_{dataset}_{selector_key}_{split}.json"
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
        num_predict = compute_num_predict(
            [qwen_hyp, whisperx_hyp, parakeet_hyp, wav2vec2_hyp]
        )

        best_hyp = ollama_select(
            client, selector_model, qwen_hyp, whisperx_hyp, parakeet_hyp,
            wav2vec2_hyp, auto_rules, num_predict
        )

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":                   ref,
            "hyp":                   best_hyp,
            "qwen_base":             qwen_hyp,
            "whisperx_hyp":          whisperx_hyp,
            "parakeet_hyp":          parakeet_hyp,
            "wav2vec2_hyp":          wav2vec2_hyp,
            "sample_WER":            sample_wer_val,
            "severity":              None,
            "auto_rules_used":       auto_rules,
            "dataset_index":         idx,
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
        "selector":         selector_key,
        # FIX: was "context_v2" - now matches this script's actual
        # identity in the grid, consistent with the filename/folder.
        "approach":         "anchored_correction_v2",
        "strategy":         "anchored_correction",
        "context_condition": "v2",
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "split":            split,
        "auto_rules_used":  auto_rules,
        "subset_indices":   indices,
        "corpus_wer":       corpus_wer,
        "num_samples":      len(valid),
        "samples":          results,
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
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--gap",         type=float, default=1.0,
                        help="Min gap (pp) for eval-suite rule generation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} split={args.split}")
        print(f"Auto-generated rules:\n{auto_rules}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Using auto-generated rules:\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()
