"""
scripts/combination/context/run_context_selector_v2_whisperx_probscore.py

Context V2 selector with PROBSCORE prompt variant.

Instead of asking for confidence 1-5, asks for probability 0.0-1.0
that each sentence is correctly transcribed.

Based on Yang et al. (2024) who found "probscore" formulation
("probability that your answer is correct") improves calibration
over "confscore" ("how confident you are") for both small and large LLMs.

Output: results/combinations_v2judge/context_v2_whisperx_probscore_meaning_meaning_meaning/

Usage:
    python scripts/combination/context/run_context_selector_v2_whisperx_probscore.py --dataset commonvoice
"""

import json
import re
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
from src.rules import build_rules_text
from src import selector as _sel

_sel.SUBSET_FILES[("whisper", "commonvoice")]      = "whisperx_commonvoice_sub150.json"
_sel.SUBSET_FILES[("whisper", "edacc")]            = "whisperx_edacc_sub150.json"
_sel.SUBSET_FILES[("whisper", "english_dialects")] = "whisperx_english_dialects_sub150.json"

OUTPUT_DIR  = "results/combinations_v2judge/context_v2_whisperx_probscore_meaning"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

# ── Probscore prompt ───────────────────────────────────────────────────────────
# Differs from confscore variant in two ways:
#   1. Asks for probability 0.0-1.0 instead of confidence 1-5
#   2. Frames as "probability this sentence is correctly transcribed"
#      rather than "how confident you are"
# Based on Yang et al. (2024): probscore formulation improves calibration
# for both small and large LLMs.

SELECTOR_PROMPT_TEMPLATE = """You are a transcript editor for Scottish English police interview audio.

Three ASR models transcribed the same audio. Your job is to produce the single most accurate transcript by choosing the best words from the three versions.

TRANSCRIPT A (whisper — most reliable, use as your base):
{whisper}

TRANSCRIPT B (qwen):
{qwen}

TRANSCRIPT C (parakeet):
{parakeet}

SELECTION RULE:
Go word by word through Transcript A. Keep each word from A unless both B and C agree on a different word — in that case use the word B and C agree on.

RELIABILITY HINTS:
- For negation words, prefer qwen if it disagrees with the others.
- For Scottish dialect words, prefer qwen if it disagrees with the others.

OUTPUT RULES:
- Output only the final transcript text — no labels, no notes, no explanations.
- Do not write anything in parentheses.
- Do not mention which model you chose or why.
- The output is what a human reviewer will read as the final transcript.

After the transcript, for each sentence estimate the probability from 0.00 to 1.00 that the final sentence preserves the spoken meaning, based only on the evidence available in the three ASR transcripts.
Use model agreement and disagreement as evidence. Pay particular attention to disagreements involving negation, names, numbers, dates, actions, timing, or speaker responsibility, since these are most likely to alter meaning.
Use the full range from 0.00 to 1.00 — do not default to values near 1.00.

Return ONLY this format:
TRANSCRIPT:
<transcript here — with proper punctuation, sentences ending in . or ? or !>

PROBABILITY:
1 | <probability 0.00-1.00> | <sentence 1 — one sentence only>
2 | <probability 0.00-1.00> | <sentence 2 — one sentence only>
...

IMPORTANT: The PROBABILITY section must have exactly one line per sentence.
Never put multiple sentences on one PROBABILITY line."""


def parse_selector_response(raw: str):
    """Parse combined selector response into (transcript, sentence_confidences)."""
    transcript           = raw
    sentence_confidences = []

    if "TRANSCRIPT:" in raw and "PROBABILITY:" in raw:
        parts           = raw.split("PROBABILITY:")
        transcript      = parts[0].replace("TRANSCRIPT:", "").strip()
        prob_part       = parts[1].strip() if len(parts) > 1 else ""

        sent_pos = 0
        for line in prob_part.split("\n"):
            line = line.strip()
            if not line:
                continue
            # match: N | 0.85 | sentence text
            match = re.match(r"^(\d+)\s*\|\s*([0-9.]+)\s*\|\s*(.+)$", line)
            if match:
                sent_pos += 1
                prob = float(match.group(2))
                prob = max(0.0, min(1.0, prob))  # clamp to [0, 1]
                sentence_confidences.append({
                    "idx":        sent_pos,
                    "score":      None,        # no discrete score in probscore variant
                    "confidence": round(prob, 4),
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
    print(f"\n── {dataset} | probscore selector={selector_key} ──")

    qwen_samples     = json.load(open(CANONICAL_FILES[("qwen",     dataset)]))["samples"]
    whisper_samples  = json.load(open(CANONICAL_FILES[("whisper",  dataset)]))["samples"]
    parakeet_samples = json.load(open(CANONICAL_FILES[("parakeet", dataset)]))["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"context_v2_probscore_{dataset}_{selector_key}_sub150.json")

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
            results.append({
                "ref": ref, "hyp": None, "qwen_verdict_p2": None,
                "sample_WER": None, "skipped": True, "dataset_index": idx,
            })
            continue

        best_hyp, sentence_confidences = ollama_select(
            client, selector_model, qwen_hyp, whisper_hyp, parakeet_hyp, auto_rules
        )
        time.sleep(0.05)

        if best_hyp is None:
            results.append({
                "ref": ref, "hyp": None, "qwen_verdict_p2": None,
                "sample_WER": None, "error": True, "dataset_index": idx,
            })
            continue

        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":                  ref,
            "hyp":                  best_hyp,
            "qwen_base":            qwen_hyp,
            "whisper_hyp":          whisper_hyp,
            "parakeet_hyp":         parakeet_hyp,
            "sample_WER":           sample_wer_val,
            "sentence_confidences": sentence_confidences,
            "qwen_verdict_p2":      verdict,
            "auto_rules_used":      auto_rules,
            "dataset_index":        idx,
            "prompt_variant":       "probscore",
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results},
                          f, indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{len(indices)} done")

    valid      = [r for r in results
                  if not r.get("skipped") and not r.get("error")
                  and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid],
    ) if valid else None
    mar = sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid) if valid else None

    output = {
        "selector":                selector_key,
        "approach":                "context_v2_probscore",
        "prompt_variant":          "probscore",
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

    return {"dataset": dataset, "wer": corpus_wer, "mar": mar, "n": len(valid)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--selector",    default="qwen",        choices=list(OLLAMA_MODELS.keys()))
    parser.add_argument("--gap",         type=float, default=1.0)
    parser.add_argument("--max-samples", type=int,   default=None)
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)
    client     = Client(host=OLLAMA_HOST)

    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return

    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Prompt variant: probscore (probability 0.0-1.0)")
    print(f"Auto-generated rules:\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()