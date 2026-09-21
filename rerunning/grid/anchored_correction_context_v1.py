"""
rerunning/grid/anchored_correction_context_v1.py

PHASE 1 of 2: Context-aware profile-guided selector V1 — hand-written
error-pattern rules. Uses Qwen3-ASR as anchor, WhisperX and Parakeet as
supporting models. Selector calls only - no severity judging here.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing_concurrent.py --files <output file from this script>

FIX (this version): filename pattern was still "context_v1_{...}",
left over from the original context_v1.py this was adapted from - only
NEW_OUTPUT_DIR had been updated to the new grid folder, not the
filename itself. This caused the folder to be correct
(writeup_results/grid/anchored_correction_v1/) but the file inside it
to be named context_v1_{dataset}_{selector}_{split}.json instead of
anchored_correction_v1_{dataset}_{selector}_{split}.json - so Phase 2
commands built by guessing the "expected" filename (matching the
script's own name) failed with FileNotFoundError, since the real saved
file used the old pattern. Filename and "approach" field now both
correctly say "anchored_correction_v1", matching the rest of the grid
(selection_naive, unanchored_fusion_context_v1, etc.) and the folder
name.

NOT CHANGED - NEEDS YOUR REVIEW: the rule content itself (the
"Whisper (B) is more reliable on named entities" line, and the other
hand-written error-pattern rules) is left exactly as written.

Usage:
    python rerunning/grid/anchored_correction_context_v1.py --dataset commonvoice --split dev
    python rerunning/grid/anchored_correction_context_v1.py --dataset edacc --split full --selector gemma4
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

NEW_OUTPUT_DIR = "writeup_results/clean_grid_guidance_rerun/anchored_correction_v1"
OLLAMA_HOST    = "http://localhost:11434"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

# NOTE: rule content unchanged - review per the discussion above before editing.
SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given four transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point, model: Qwen):
{qwen}

TRANSCRIPT B (model: WhisperX):
{whisperx}

TRANSCRIPT C (model: Parakeet):
{parakeet}

TRANSCRIPT D (model: wav2vec2):
{wav2vec2}

Your task is to return Transcript A with targeted corrections where needed.

GENERAL RULE — applies unless a specific rule below overrides it:
Only change a word or short phrase in Transcript A when at least two of Transcripts B, C, and D disagree with A and agree with each other on the same alternative. If fewer than two supporting transcripts agree on an alternative, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

HUMAN-WRITTEN ERROR GUIDANCE:
- NAMED ENTITIES: names of people, places, and organisations are especially vulnerable to plausible substitutions. Replace A's named-entity form only when at least two of B, C, and D agree on the same alternative; do not trust a source model automatically.
- PROFANITY AND INFORMAL EXPRESSIONS: if at least two of B, C, and D preserve the same original expression, use that expression rather than silently sanitising or normalising it.
- NEGATIONS: If A drops or changes a negation and at least two of B, C, and D preserve the same negation, correct A.
- NUMBERS: If A contains a different number and at least two of B, C, and D agree on the same alternative, correct A.
- SCOTTISH DIALECT WORDS: If A normalises a Scottish dialect word and at least two of B, C, and D preserve the same dialect form, use the dialect form.
- PRONOUNS: If A changes a pronoun and at least two of B, C, and D agree on the same alternative, correct A.

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


def ollama_select(client, model_name, qwen_hyp, whisperx_hyp, parakeet_hyp,
                   wav2vec2_hyp, num_predict, retries=2):
    prompt = SELECTOR_PROMPT.format(
        qwen=qwen_hyp, whisperx=whisperx_hyp,
        parakeet=parakeet_hyp, wav2vec2=wav2vec2_hyp,
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

    print(f"\n── {dataset} | anchored correction V1 (PHASE 1: selector only) selector={selector_key} split={split} ──")

    qwen_samples     = get_indexed_samples("qwen", dataset)
    whisperx_samples = get_indexed_samples("whisperx", dataset)
    parakeet_samples = get_indexed_samples("parakeet", dataset)
    wav2vec2_samples = get_indexed_samples("wav2vec2", dataset)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

    # FIX: was "context_{dataset}_{selector_key}_{split}.json" - now
    # matches the approach name and folder name consistently.
    filename = f"anchored_correction_v1_{dataset}_{selector_key}_{split}.json"
    new_output_path = os.path.join(NEW_OUTPUT_DIR, filename)

    if os.path.exists(new_output_path) and not rerun:
        with open(new_output_path) as f:
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
            wav2vec2_hyp, num_predict
        )

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
            "wav2vec2_hyp":   wav2vec2_hyp,
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
        # FIX: was "context_v1" - now matches this script's actual
        # identity in the grid, consistent with the filename/folder.
        "approach":       "anchored_correction_v1",
        "strategy":       "anchored_correction",
        "context_condition": "v1",
        "guidance_source": "human_written_linguistic_error_guidance",
        "phase":          "selector_only - severity not yet judged",
        "dataset":        dataset,
        "split":          split,
        "subset_indices": indices,
        "corpus_wer":     corpus_wer,
        "num_samples":    len(valid),
        "samples":        results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2:")
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
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
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()
