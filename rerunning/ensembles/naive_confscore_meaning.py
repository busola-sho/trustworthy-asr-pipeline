"""
rerunning/ensembles/naive_confscore_meaning.py

Naive fusion (locked best technique: gemma4 selector, all 4 ASR models,
rotated/anonymized transcripts) EXTENDED with a "meaning-focused"
per-sentence CONFIDENCE score (1-5 scale) - requested by Miri to test
alongside the plain confscore/probscore variants.

WHAT MAKES THIS DIFFERENT FROM naive_confscore.py (not just relabeled):
  1. The confidence framing explicitly asks "does this sentence preserve
     the spoken MEANING" rather than a generic "how much did the models
     agree" - directly matching your severity metric's actual target.
  2. Explicitly directs attention to the categories most likely to
     silently alter meaning even under high surface agreement: negation,
     names, numbers, dates, actions, timing, speaker responsibility.
  3. Stricter anti-collapsing instruction (exactly one line per sentence
     in the CONFIDENCE block, never merged) - the original context_v2
     confscore_meaning script needed this explicitly, suggesting gemma4
     (or its era's selector) sometimes collapsed multiple sentences onto
     one line without it.
  4. Explicit "no meta-commentary, silent corrections only" instruction
     for the transcript itself - prevents the model leaking correction
     notes/parentheticals into what's supposed to be pure transcript text.

CONCURRENT: same Phase A/B pattern as the rest of your pipeline.

Usage:
    python rerunning/ensembles/naive_confscore_meaning.py --dataset commonvoice --split dev
    python rerunning/ensembles/naive_confscore_meaning.py --dataset shetland --split full
"""

import json
import os
import re
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

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive_confscore_meaning"
OLD_OUTPUT_DIR = "results/combinations_v2judge/naive_confscore_meaning"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]

DEFAULT_MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))

# Naive's original selection task, unchanged (no context_v2-style fixed
# model roles or reliability hints - that machinery doesn't carry over)
# + the MEANING-focused confidence framing from the original
# confscore_meaning script, adapted from 3 fixed-role transcripts to 4
# anonymized/rotated ones.
SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps
- Add any explanations, correction notes, or parenthetical comments - if you correct a word, write the corrected word silently; the transcript must contain ONLY the spoken words

After producing the combined transcript, break it into sentences and rate each sentence on how likely it is that it preserves the spoken meaning, based only on the evidence available in the four transcripts.

Confidence scale:
5 = models fully agree - meaning almost certainly preserved
4 = minor differences only - meaning likely preserved
3 = some disagreement on words that could affect meaning - uncertain
2 = significant disagreement on meaningful content - meaning possibly altered
1 = models strongly disagree on meaningful content - meaning likely altered

Pay particular attention to disagreements involving negation, names, numbers, dates, actions, timing, or speaker responsibility - these are most likely to alter meaning even when overall agreement appears high.

Return ONLY this format - no explanation, no preamble:
TRANSCRIPT:
<your combined transcript here>

CONFIDENCE:
1 | <score 1-5> | <sentence 1>
2 | <score 1-5> | <sentence 2>
...

IMPORTANT: The CONFIDENCE section must have exactly one line per sentence.
Never put multiple sentences on one CONFIDENCE line.
Never put the whole transcript on one CONFIDENCE line."""


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
    """
    Larger buffer than naive.py's plain version, since the response now
    also includes the per-sentence PROBABILITY block on top of the
    transcript itself - roughly transcript length again, plus overhead
    for the "N | prob | sentence" formatting per line.
    """
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 2.2) + 150
    return max(400, min(estimated, 3000))


def parse_selector_response(raw: str):
    """
    Parses "TRANSCRIPT:\\n<text>\\nCONFIDENCE:\\n1 | 4 | sentence" into
    (transcript, sentence_confidences). Confidence is normalized to 0-1
    (score/5.0) so it's on the same scale as probscore's output for
    direct comparison - "score" keeps the raw 1-5 integer too.
    """
    transcript = raw
    sentence_confidences = []

    if "TRANSCRIPT:" in raw and "CONFIDENCE:" in raw:
        parts = raw.split("CONFIDENCE:")
        transcript = parts[0].replace("TRANSCRIPT:", "").strip()
        conf_part = parts[1].strip() if len(parts) > 1 else ""

        sent_pos = 0
        for line in conf_part.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s*\|\s*([1-5])\s*\|\s*(.+)$", line)
            if match:
                sent_pos += 1
                score = int(match.group(2))
                sentence_confidences.append({
                    "idx":        sent_pos,
                    "score":      score,
                    "confidence": round(score / 5.0, 4),
                    "sentence":   match.group(3).strip(),
                })
    elif "TRANSCRIPT:" in raw:
        transcript = raw.replace("TRANSCRIPT:", "").strip()

    return transcript, sentence_confidences


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


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False,
                 split="dev", max_workers=None):
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive_confscore_meaning (PHASE 1: selector + confscore_meaning, CONCURRENT) "
          f"selector={selector_key} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"naive_confscore_meaning_{dataset}_{selector_key}sel_{split}.json"
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
    work_items = []
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

    raw_responses = run_concurrent(work_items, _worker, max_workers=max_workers, progress_every=10)

    # ── Reassemble in original index order, parsing the probscore response ──
    call_results = {item[0]: (item, raw) for item, raw in zip(work_items, raw_responses)}

    for idx in remaining_indices:
        if idx in skip_slots:
            results.append(skip_slots[idx])
            continue

        item, raw = call_results[idx]
        _, ref, model_order, hyp_by_model, _ = item

        if raw is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp, sentence_confidences = parse_selector_response(raw)

        if not best_hyp:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "could not parse TRANSCRIPT block",
                            "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":                  ref,
            "hyp":                  best_hyp,
            "source_hyps":          hyp_by_model,
            "model_order":          model_order,
            "sample_WER":           sample_wer_val,
            "sentence_confidences": sentence_confidences,
            "severity":             None,
            "dataset_index":        idx,
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

    n_missing_probs = sum(1 for r in valid if not r.get("sentence_confidences"))

    output = {
        "selector":         selector_key,
        "approach":         "naive_confscore_meaning",
        "prompt_variant":   "confscore",
        "asr_models":       ASR_MODELS,
        "phase":            "selector_only - severity not yet judged",
        "dataset":          dataset,
        "split":            split,
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
    if n_missing_probs:
        print(f"  WARNING: {n_missing_probs}/{len(valid)} samples had no parsed "
              f"sentence_confidences (selector didn't follow the CONFIDENCE format)")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing_concurrent.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
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
