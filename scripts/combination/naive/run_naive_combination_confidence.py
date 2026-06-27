"""
run_naive_combination_confidence.py

Naive combination + word-level confidence flagging. Whisper and Parakeet
transcripts have low-confidence words marked with [LOW-CONF: word] before
being passed to the selector. Qwen and wav2vec2 pass through unchanged
(no confidence data available for these models).

Isolates whether confidence flagging helps independently of any model-trust
rules — naive has no error-pattern rules at all.

Usage:
    python scripts/combination/run_naive_combination_confidence.py --dataset commonvoice --threshold 0.5
    python scripts/combination/run_naive_combination_confidence.py --dataset edacc --threshold 0.8
"""

import json
import os
import argparse
import time
from jiwer import wer
from ollama import Client

from src.judge import normalise, ollama_mar
from src.selector import (
    get_subset_indices, CANONICAL_FILES, OLLAMA_MODELS,
    DATASET_SIZES, check_selector_available,
    load_subset_by_sample_index, get_transcript_for_selector,
    count_flagged_words, strip_lowconf_markers,
    DEFAULT_CONF_THRESHOLD,
)

OUTPUT_DIR  = "results/combinations_v2judge/naive_confidence"
OLLAMA_HOST = "http://localhost:11434"
ASR_MODELS  = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

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


def run_dataset(dataset, selector_key, client, threshold, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | naive + confidence selector={selector_key} t={threshold} ──")

    qwen_samples     = json.load(open(CANONICAL_FILES[("qwen",     dataset)]))["samples"]
    wav2vec2_samples = json.load(open(CANONICAL_FILES[("wav2vec2", dataset)]))["samples"]
    whisper_by_idx   = load_subset_by_sample_index("whisper",  dataset)
    parakeet_by_idx  = load_subset_by_sample_index("parakeet", dataset)

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresh_str  = f"t{threshold:.2f}".replace(".", "")
    output_path = os.path.join(OUTPUT_DIR,
                               f"naive_conf_{dataset}_{selector_key}_{thresh_str}_sub150.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    for pos in range(start_from, len(indices)):
        idx          = indices[pos]
        ref          = qwen_samples[idx]["ref"]
        qwen_hyp     = qwen_samples[idx]["hyp"]
        wav2vec2_hyp = wav2vec2_samples[idx]["hyp"]

        whisper_sample  = whisper_by_idx.get(idx)
        parakeet_sample = parakeet_by_idx.get(idx)

        if whisper_sample is None or parakeet_sample is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing subset sample",
                            "dataset_index": idx})
            continue

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        whisper_text  = get_transcript_for_selector(
            whisper_sample["hyp"], whisper_sample.get("segments"), threshold)
        parakeet_text = get_transcript_for_selector(
            parakeet_sample["hyp"], parakeet_sample.get("segments"), threshold)

        hyps = [qwen_hyp, whisper_text, parakeet_text, wav2vec2_hyp]
        raw  = ollama_select(client, selector_model, hyps)
        time.sleep(0.1)

        if raw is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp       = strip_lowconf_markers(raw)
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.1)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "source_hyps": {
                "qwen":              qwen_hyp,
                "whisper_flagged":   whisper_text,
                "parakeet_flagged":  parakeet_text,
                "wav2vec2":          wav2vec2_hyp,
            },
            "n_flagged_whisper":    count_flagged_words(whisper_sample.get("segments"),  threshold),
            "n_flagged_parakeet":   count_flagged_words(parakeet_sample.get("segments"), threshold),
            "qwen_verdict_p2":      verdict,
            "sample_WER":           sample_wer_val,
            "dataset_index":        idx,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{len(indices)} done")

    valid      = [r for r in results if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "approach":                "naive_confidence",
        "dataset":                 dataset,
        "confidence_threshold":    threshold,
        "subset_indices":          indices,
        "corpus_wer":              corpus_wer,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    mar_str = f"{mar*100:.2f}%"        if mar        is not None else "—"
    print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")
    return {"dataset": dataset, "selector": selector_key,
            "wer": corpus_wer, "mar": mar, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--threshold",   type=float,            default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--max-samples", type=int,              default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} threshold={args.threshold}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client, args.threshold,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()