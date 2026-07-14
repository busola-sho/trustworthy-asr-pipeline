"""
run_context_selector_v3.py

Context-aware V3 — builds on V2 (auto-generated eval-suite rules) with
confidence flagging now applied to ALL three models: Qwen, Whisper, and
Parakeet. Previously only Whisper and Parakeet were flagged; Qwen passed
through as a plain anchor. Now Qwen's own uncertainty is visible too.

Usage:
    python scripts/combination/run_context_selector_v3.py --dataset commonvoice --threshold 0.8
    python scripts/combination/run_context_selector_v3.py --dataset edacc --threshold 0.5 --gap 1.0
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
from src.rules import build_rules_text

OUTPUT_DIR  = "results/combinations_v2judge/context_v3"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

SELECTOR_PROMPT_TEMPLATE = """You are correcting an ASR transcript. You are given three transcripts of the same audio from different models.

All three transcripts may include inline [LOW-CONF: word] markers — these indicate words that model itself was uncertain about. Treat words marked this way as MORE likely to be wrong. Do NOT include the [LOW-CONF: ...] markers in your final output — write the plain word only.

TRANSCRIPT A (base — use this as your starting point, model: qwen):
{qwen}

TRANSCRIPT B (model: whisper):
{whisper}

TRANSCRIPT C (model: parakeet):
{parakeet}

Your task: return Transcript A with targeted corrections where needed.

GENERAL RULE — applies to all words except where a specific rule below overrides it:
Only change a word if BOTH B and C disagree with A and agree with each other on the same alternative. If only one of B or C disagrees with A, keep A unchanged. Ignore capitalisation and punctuation differences when checking agreement.
If a disagreeing word in any transcript is marked [LOW-CONF: ...], treat that disagreement as weaker evidence.
If a word in Transcript A is marked [LOW-CONF: ...] AND both B and C agree on a different word, strongly prefer switching — Qwen itself is uncertain here.

MODEL-SPECIFIC RELIABILITY RULES (derived from measured error rates across all benchmark datasets):
{auto_rules}

Do NOT paraphrase, reorder, or restructure sentences.
Do NOT add words that appear in none of the three transcripts.
Do NOT include any [LOW-CONF: ...] markers in your output.

Return ONLY the corrected transcript. No explanation, no labels, no preamble."""


def ollama_select(client, model_name, qwen_text, whisper_text, parakeet_text, auto_rules):
    prompt = SELECTOR_PROMPT_TEMPLATE.format(
        qwen=qwen_text, whisper=whisper_text,
        parakeet=parakeet_text, auto_rules=auto_rules,
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


def run_dataset(dataset, selector_key, client, auto_rules, threshold,
                max_samples=None, rerun=False):
    selector_model = OLLAMA_MODELS[selector_key]

    print(f"\n── {dataset} | context V3 selector={selector_key} t={threshold} ──")

    qwen_by_idx     = load_subset_by_sample_index("qwen",     dataset)
    whisper_by_idx  = load_subset_by_sample_index("whisper",  dataset)
    parakeet_by_idx = load_subset_by_sample_index("parakeet", dataset)

    with open(CANONICAL_FILES[("qwen", dataset)]) as f:
        qwen_full = json.load(f)["samples"]

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    thresh_str  = f"t{threshold:.2f}".replace(".", "")
    output_path = os.path.join(OUTPUT_DIR,
                               f"context_v3_{dataset}_{selector_key}_{thresh_str}_sub150.json")

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
        idx = indices[pos]
        ref = qwen_full[idx]["ref"]

        qwen_sample     = qwen_by_idx.get(idx)
        whisper_sample  = whisper_by_idx.get(idx)
        parakeet_sample = parakeet_by_idx.get(idx)

        if any(s is None for s in [qwen_sample, whisper_sample, parakeet_sample]):
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing subset sample",
                            "dataset_index": idx})
            continue

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "skipped": True, "dataset_index": idx})
            continue

        qwen_text     = get_transcript_for_selector(
            qwen_sample["hyp"], qwen_sample.get("segments"), threshold)
        whisper_text  = get_transcript_for_selector(
            whisper_sample["hyp"], whisper_sample.get("segments"), threshold)
        parakeet_text = get_transcript_for_selector(
            parakeet_sample["hyp"], parakeet_sample.get("segments"), threshold)

        raw = ollama_select(client, selector_model, qwen_text,
                            whisper_text, parakeet_text, auto_rules)
        time.sleep(0.05)

        if raw is None:
            results.append({"ref": ref, "hyp": None, "qwen_verdict_p2": None,
                            "sample_WER": None, "error": True, "dataset_index": idx})
            continue

        best_hyp       = strip_lowconf_markers(raw)
        sample_wer_val = wer(normalise(ref), normalise(best_hyp))
        verdict        = ollama_mar(client, ref, best_hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":                  ref,
            "hyp":                  best_hyp,
            "qwen_hyp_flagged":     qwen_text,
            "whisper_hyp_flagged":  whisper_text,
            "parakeet_hyp_flagged": parakeet_text,
            "n_flagged_qwen":       count_flagged_words(qwen_sample.get("segments"),     threshold),
            "n_flagged_whisper":    count_flagged_words(whisper_sample.get("segments"),  threshold),
            "n_flagged_parakeet":   count_flagged_words(parakeet_sample.get("segments"), threshold),
            "sample_WER":           sample_wer_val,
            "qwen_verdict_p2":      verdict,
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
        "approach":                "context_v3",
        "dataset":                 dataset,
        "confidence_threshold":    threshold,
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
    parser.add_argument("--threshold",   type=float,            default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--gap",         type=float,            default=1.0)
    parser.add_argument("--max-samples", type=int,              default=None)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    auto_rules = build_rules_text(min_gap_pp=args.gap, save=False)

    if args.dry_run:
        print(f"[DRY RUN] dataset={args.dataset} selector={args.selector} threshold={args.threshold}")
        print(f"Auto rules:\n{auto_rules}")
        return

    client = Client(host=OLLAMA_HOST)
    if not check_selector_available(client, args.selector):
        print(f"ERROR: {OLLAMA_MODELS[args.selector]} not pulled.")
        return
    print(f"Ollama connected. Selector: {OLLAMA_MODELS[args.selector]}")
    print(f"Using auto-generated rules:\n{auto_rules}\n")

    run_dataset(args.dataset, args.selector, client, auto_rules, args.threshold,
                max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()