"""
run_context_selector.py

Context-aware profile-guided selector V1 — hand-written error-pattern rules.
Uses Qwen3-ASR as anchor, Whisper and Parakeet as supporting models.
Restricted to the 150-sample subset (seed=42).
Output saved to results/combinations_v2judge/context/.

Usage:
    python scripts/combination/run_context_selector.py --dataset commonvoice
    python scripts/combination/run_context_selector.py --dataset edacc --selector gemma2
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
)

OUTPUT_DIR  = "results/combinations_v2judge/context"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

TRANSCRIPT A (base — use this as your starting point):
{qwen}

TRANSCRIPT B (Whisper):
{whisper}

TRANSCRIPT C (Parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except named entities:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.

NAMED ENTITY RULE — applies to people's names, place names, organisations:
Whisper (B) is more reliable on named entities. If B has a different named entity than A, consider switching — BUT only if C does not agree with A (case-insensitive). If C agrees with A on the named entity, keep A.

ADDITIONAL KNOWN ERROR PATTERNS:
- PROFANITY AND INFORMAL EXPRESSIONS: A sometimes self-censors mild profanity (e.g. "shit-scared"→"scared", "bloody"→"body", "sweet F all"→"sweetie fall") — if B and C have the original expression, restore it.
- NEGATIONS: If A drops or changes a negation and B and C preserve it — this is critical, switch.
- NUMBERS: If A has a different number than B and C agree on — switch.
- SCOTTISH DIALECT WORDS: If B or C preserve a Scottish dialect word that A has normalised (e.g. "wee", "wisnae", "dinnae", "cannae", "braw", "aboot", "carry-out", "noo") and both agree — preserve the dialect word.
- PRONOUNS: If A changes a pronoun (I/you/we/she/they) and B and C agree on the original — switch.

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def ollama_select(client, model_name, qwen_hyp, whisper_hyp, parakeet_hyp):
    prompt = SELECTOR_PROMPT.format(
        qwen=qwen_hyp, whisper=whisper_hyp, parakeet=parakeet_hyp
    )
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"  ERROR (selector): {e}")
        return None


def run_dataset(dataset, selector_key, client, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V1 selector={selector_key} ──")

    qwen_samples    = json.load(open(CANONICAL_FILES[("qwen",     dataset)]))["samples"]
    whisper_samples = json.load(open(CANONICAL_FILES[("whisper",  dataset)]))["samples"]
    parakeet_samples= json.load(open(CANONICAL_FILES[("parakeet", dataset)]))["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"context_{dataset}_{selector_key}_sub150.json")

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
        whisper_hyp  = whisper_samples[idx]["hyp"]
        parakeet_hyp = parakeet_samples[idx]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        best_hyp = ollama_select(client, selector_model, qwen_hyp, whisper_hyp, parakeet_hyp)
        time.sleep(0.05)

        if best_hyp is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":             ref,
            "hyp":             best_hyp,
            "qwen_base":       qwen_hyp,
            "whisper_hyp":     whisper_hyp,
            "parakeet_hyp":    parakeet_hyp,
            "sample_WER":      sample_wer_val,
            "qwen_verdict_p2": verdict,
            "dataset_index":   idx,
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
        "approach":                "context_v1",
        "dataset":                 dataset,
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")

    run_dataset(args.dataset, args.selector, client,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()