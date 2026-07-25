"""
rerunning/ensembles/naive_confidence_inline.py

PHASE 1 of 2: Naive combination + word-level confidence flagging using
INLINE markers - built specifically to compare against
naive_confidence.py's SEPARATE/structured metadata approach.

Same reasoning as context_v2_confidence_inline.py: inline markers sit at
the exact occurrence of a flagged word in the running sentence, so
repeated words are naturally disambiguated by position - the separate
design needed an explicit position-index fix to recover that. Run both
and compare, rather than assuming one representation is better.

Once this finishes, run PHASE 2:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

Same underlying fixes as naive_confidence.py: WhisperX swap, per-model
percentile thresholds, two-pass split, tag-only skip, dual write,
dynamic num_predict, keep_alive, order rotation. --split {dev,test,full}
replaces --full - defaults to "dev" for iteration.

FIXED (this pass): think=False added to the Ollama call - without it,
thinking-capable selectors (e.g. gemma4, the locked selector per
selector_ablation.py) spend the whole generation budget on hidden
reasoning tokens and return an EMPTY message.content, silently producing
hyp="" and sample_WER=1.0 for every sample rather than an error.
Invisible while the default selector was qwen2.5 (not a thinking model).
--selector default updated to gemma4 per the ablation lock
(mean_severity=0.85, mean_wer=9.26%, compliance=100%).

Usage:
    python rerunning/ensembles/naive_confidence_inline.py --dataset commonvoice --split dev --percentile 20
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

NEW_OUTPUT_DIR = "writeup_results/ensembles/naive_confidence_inline"
OLD_OUTPUT_DIR = "results/combinations_v2judge/naive_confidence_inline"
OLLAMA_HOST    = "http://localhost:11434"
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
CONFIDENCE_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]   # all 4 now produce confidence scores
DEFAULT_PERCENTILE = 20

SELECTOR_PROMPT = """You are given four ASR transcripts of the same spoken audio.

Some transcripts include inline [LOW-CONF: word] markers — these indicate words the model itself was uncertain about. Treat such words as MORE likely to be wrong when deciding what to combine. Do NOT include the [LOW-CONF: ...] markers in your final output — write the plain word only.

Your task is to construct the most accurate transcript by selecting the best words and phrases from the four options. You may:
- Select one transcript verbatim
- Swap individual words or short phrases between transcripts where one is clearly more accurate (e.g. a correct name, number, or dialect word)

You MUST NOT:
- Paraphrase or rewrite sentences
- Add any words not present in any of the four transcripts
- Change sentence structure or word order beyond individual word swaps
- Include any [LOW-CONF: ...] markers in your output

Return only the final transcript, nothing else."""


def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def compute_percentile_thresholds(dataset: str, percentile: int) -> dict:
    thresholds = {}
    for model in CONFIDENCE_MODELS:
        path = find_canonical_file(model, dataset)
        samples = load_samples(path)
        scores = [
            seg["confidence"]
            for s in samples
            for seg in (s.get("segments") or [])
            if seg.get("confidence") is not None
        ]
        if not scores:
            thresholds[model] = 0.5
            continue
        scores.sort()
        idx = min(int(len(scores) * percentile / 100), len(scores) - 1)
        thresholds[model] = scores[idx]
        print(f"  Calibrated threshold for {model} (p{percentile}): {thresholds[model]:.3f} "
              f"(from {len(scores)} word scores)")
    return thresholds


def build_inline_transcript(hyp: str, segments: list, threshold: float) -> str:
    """Inserts [LOW-CONF: word] markers at the exact position of each
    flagged word - naturally position-exact, unlike a separate list."""
    if not segments:
        return hyp
    parts = []
    for seg in segments:
        word = seg.get("word", "")
        conf = seg.get("confidence")
        if not word:
            continue
        if conf is not None and conf < threshold:
            parts.append(f"[LOW-CONF: {word}]")
        else:
            parts.append(word)
    return " ".join(parts) if parts else hyp


def strip_lowconf_markers(text: str) -> str:
    return re.sub(r'\[LOW-CONF:\s*([^\]]+)\]', r'\1', text)


def get_rotated_order(models: list, seed_key: int) -> list:
    order = list(models)
    rng = random.Random(seed_key)
    rng.shuffle(order)
    return order


def compute_num_predict(hyps: list) -> int:
    max_words = max(len(h.split()) for h in hyps)
    estimated = int(max_words * 1.3) + 50
    return max(300, min(estimated, 2048))


def ollama_select(client, model_name, prompt_block, num_predict, retries=2):
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": SELECTOR_PROMPT},
                    {"role": "user",   "content": prompt_block},
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


def run_dataset(dataset, selector_key, client, percentile, max_samples=None,
                 rerun=False, split="dev"):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive + confidence INLINE (PHASE 1: selector only) "
          f"selector={selector_key} percentile={percentile} split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}
    thresholds = compute_percentile_thresholds(dataset, percentile)

    indices = get_indices_for_split(dataset, split)
    print(f"  Split '{split}': {len(indices)} samples")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"naive_confinline_{dataset}_{selector_key}_p{percentile}_{split}.json"
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

        hyp_by_model = {}
        for m in ASR_MODELS:
            s = samples_by_model[m]
            if m in CONFIDENCE_MODELS:
                hyp_by_model[m] = build_inline_transcript(s["hyp"], s.get("segments"), thresholds[m])
            else:
                hyp_by_model[m] = s["hyp"]

        model_order = get_rotated_order(ASR_MODELS, seed_key=idx)
        prompt_block = "\n".join(
            f"Transcript {i+1}: {hyp_by_model[m]}\n" for i, m in enumerate(model_order)
        )
        num_predict = compute_num_predict(list(hyp_by_model.values()))

        raw = ollama_select(client, selector_model, prompt_block, num_predict)

        if raw is None:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp = strip_lowconf_markers(raw)
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))

        results.append({
            "ref":               ref,
            "hyp":               best_hyp,
            "source_hyps_flagged": hyp_by_model,
            "model_order":       model_order,
            "thresholds_used":   thresholds,
            "sample_WER":        sample_wer_val,
            "severity":          None,
            "dataset_index":     idx,
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
        "selector":             selector_key,
        "approach":             "naive_confidence_inline",
        "asr_models":           ASR_MODELS,
        "phase":                "selector_only - severity not yet judged",
        "dataset":              dataset,
        "split":                split,
        "percentile":           percentile,
        "thresholds_used":      thresholds,
        "subset_indices":       indices,
        "corpus_wer":           corpus_wer,
        "num_samples":          len(valid),
        "samples":              results,
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
    parser.add_argument("--selector",    default="gemma4",      choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--percentile",  type=int,              default=DEFAULT_PERCENTILE)
    parser.add_argument("--max-samples", type=int,              default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} "
              f"percentile={args.percentile} split={args.split}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, args.percentile,
                max_samples=args.max_samples, rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()