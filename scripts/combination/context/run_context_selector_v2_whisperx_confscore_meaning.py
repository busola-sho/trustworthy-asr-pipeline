"""
run_context_selector_v2.py

Context-aware profile-guided selector V2 — auto-generated error-pattern rules
from the automated eval suite. Uses Qwen3-ASR as anchor, Whisper and Parakeet
as supporting models. Restricted to the 150-sample subset (seed=42).
Output saved to results/combinations_v2judge/context_v2/.

Usage:
    python scripts/combination/run_context_selector_v2.py --dataset commonvoice
    python scripts/combination/run_context_selector_v2.py --dataset edacc --gap 0.5
"""

import json
import re
import os
import argparse
import time
import sys
from jiwer import wer
from ollama import Client

from src.judge import normalise, ollama_mar
from src.selector import (
    get_subset_indices, CANONICAL_FILES, OLLAMA_MODELS,
    DATASET_SIZES, check_selector_available,
)

from src.rules import build_rules_text

from src import selector as _sel
_sel.SUBSET_FILES[("whisper", "commonvoice")]      = "whisperx_commonvoice_sub150.json"
_sel.SUBSET_FILES[("whisper", "edacc")]            = "whisperx_edacc_sub150.json"
_sel.SUBSET_FILES[("whisper", "english_dialects")] = "whisperx_english_dialects_sub150.json"

# OUTPUT_DIR = "results/combinations_v2judge/context_v2_whisperx"
OUTPUT_DIR = "results/combinations_v2judge/context_v2_whisperx_confscore_meaning"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript for a Scottish English police interview — a high-stakes setting where accuracy matters.

You are given three transcripts of the same audio from different models.

TRANSCRIPT A (model: whisper — strongest model, use as primary reference):
{whisper}

TRANSCRIPT B (model: qwen):
{qwen}

TRANSCRIPT C (model: parakeet):
{parakeet}

Your task:
1. Produce the best combined transcript by correcting errors where models agree against A
2. Rate each sentence on how likely it is that the final sentence preserves the spoken meaning, based only on the evidence available in the three transcripts

GENERAL RULE:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative.
If only one of B or C disagrees with A, keep A unchanged.
Ignore capitalisation and punctuation differences.

MODEL-SPECIFIC RELIABILITY RULES:
{auto_rules}

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.
Do NOT add any explanations, correction notes, or parenthetical comments.
If you correct a word, write the corrected word silently.
The transcript must contain ONLY the spoken words.

Confidence scale:
5 = models fully agree — meaning almost certainly preserved
4 = minor differences only — meaning likely preserved
3 = some disagreement on words that could affect meaning — uncertain
2 = significant disagreement on meaningful content — meaning possibly altered
1 = models strongly disagree on meaningful content — meaning likely altered

Pay particular attention to disagreements involving negation, names, numbers, dates, actions, timing, or speaker responsibility — these are most likely to alter meaning even when overall agreement appears high.

Return ONLY this format — no explanation, no preamble:
TRANSCRIPT:
<your combined transcript here>

CONFIDENCE:
1 | <score> | <sentence 1>
2 | <score> | <sentence 2>
...

IMPORTANT: The CONFIDENCE section must have exactly one line per sentence.
Never put multiple sentences on one CONFIDENCE line.
Never put the whole transcript on one CONFIDENCE line."""


def parse_selector_response(raw: str):
    """
    Parse combined selector response into (transcript, sentence_confidences).
    Expected format:
    TRANSCRIPT:
    <text>
    CONFIDENCE:
    1 | <score> | <sentence>
    """
    transcript = raw
    sentence_confidences = []

    if "TRANSCRIPT:" in raw and "CONFIDENCE:" in raw:
        parts            = raw.split("CONFIDENCE:")
        transcript_part  = parts[0].replace("TRANSCRIPT:", "").strip()
        confidence_part  = parts[1].strip() if len(parts) > 1 else ""
        transcript       = transcript_part

        sent_pos = 0
        for line in confidence_part.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d+)\s*\|\s*([1-5])\s*\|\s*(.+)$", line)
            if match:
                sent_pos += 1
                sentence_confidences.append({
                    "idx":        sent_pos,
                    "score":      int(match.group(2)),
                    "confidence": int(match.group(2)) / 5.0,
                    "sentence":   match.group(3).strip(),
                })
    elif "TRANSCRIPT:" in raw:
        transcript = raw.replace("TRANSCRIPT:", "").strip()

    return transcript, sentence_confidences


def ollama_select(client, model_name, qwen_hyp, whisper_hyp, parakeet_hyp, auto_rules):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        whisper=whisper_hyp, qwen=qwen_hyp,
        parakeet=parakeet_hyp, auto_rules=auto_rules,
    )
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_ctx": 4096},
        )
        return parse_selector_response(response.message.content.strip())
    except Exception as e:
        print(f"  ERROR (selector): {e}")
        return None, []


def run_dataset(dataset, selector_key, client, auto_rules, max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V2 selector={selector_key} ──")

    qwen_samples     = json.load(open(CANONICAL_FILES[("qwen",     dataset)]))["samples"]
    whisper_samples  = json.load(open(CANONICAL_FILES[("whisper",  dataset)]))["samples"]
    parakeet_samples = json.load(open(CANONICAL_FILES[("parakeet", dataset)]))["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"context_v2_{dataset}_{selector_key}_sub150.json")

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

        best_hyp, sentence_confidences = ollama_select(client, selector_model, qwen_hyp,
                                 whisper_hyp, parakeet_hyp, auto_rules)
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
            "sentence_confidences": sentence_confidences,
            "qwen_verdict_p2": verdict,
            "auto_rules_used": auto_rules,
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
        "approach":                "context_v2",
        "dataset":                 dataset,
        "auto_rules_used":         auto_rules,
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
    parser.add_argument("--gap",         type=float, default=1.0,
                        help="Min gap (pp) for eval-suite rule generation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector}")
        print(f"Auto-generated rules:\n{auto_rules}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Using auto-generated rules:\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()