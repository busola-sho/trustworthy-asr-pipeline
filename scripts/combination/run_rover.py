"""
run_rover.py

Segment-level ROVER combination with sentence-level confidence scoring.

Pipeline:
  1. Load 4 model transcripts + word-level confidence segments
  2. Rank models by mean utterance confidence
  3. Use spaCy on Qwen to split all models into sentence-aligned segments
  4. Run ROVER per segment (short sequences = clean DP alignment)
  5. Resolve risky disagreements per segment with Scottish-aware LLM
  6. Concatenate segment outputs directly (no reconstruction)
  7. Compute sentence-level confidence from word votes
  8. Run MAR judge on final output
  9. Save to results/combinations_v2judge/rover/

Usage:
    python scripts/combination/run_rover.py --dataset commonvoice
    python scripts/combination/run_rover.py --dataset commonvoice --max-samples 10
    python scripts/combination/run_rover.py --dataset commonvoice --no-resolver
"""

import json
import os
import argparse
import time
import numpy as np
from jiwer import wer
from ollama import Client

from src.judge import normalise, ollama_mar
from src.selector import (
    get_subset_indices, CANONICAL_FILES,
    SUBSETS_DIR, OLLAMA_MODELS,
)
from src.rover import rover_combine, risky_disagreements
from src.segmenter import segment_hypotheses
from src.auditor import audit_sentence, check_auditor_available
from src.dialect_pass import run_dialect_pass, check_dialect_pass_available

OUTPUT_DIR  = "results/combinations_v2judge/rover"
OLLAMA_HOST = "http://localhost:11434"
DATASETS    = ["commonvoice", "english_dialects", "edacc", "shetland"]

ASR_MODELS  = ["qwen", "whisper", "parakeet", "wav2vec2"]
CONF_MODELS = ["qwen", "whisper", "parakeet"]

SUBSET_MODEL_MAP = {
    "qwen":     "qwen3asr",
    "whisper":  "whisper",
    "parakeet": "parakeet",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_confidence_by_index(model_key: str, dataset: str) -> dict:
    fname_prefix = SUBSET_MODEL_MAP.get(model_key, model_key)
    for suffix in ["_sub150", "_sub100", ""]:
        fname = f"{fname_prefix}_{dataset}{suffix}.json"
        path  = os.path.join(SUBSETS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return {
                s["sample_index"]: s.get("segments", [])
                for s in data.get("samples", [])
                if s.get("sample_index") is not None
            }
    return {}


def mean_confidence(segments: list) -> float:
    if not segments:
        return 0.5
    confs = [s["confidence"] for s in segments if s.get("confidence") is not None]
    return float(np.mean(confs)) if confs else 0.5


def rank_models(conf_data: dict, idx: int) -> list:
    """Rank models by mean utterance confidence. wav2vec2 always last."""
    scores = {}
    for model in CONF_MODELS:
        segs = conf_data.get(model, {}).get(idx, [])
        scores[model] = mean_confidence(segs)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    ordered = [m for m, _ in ranked]
    if "wav2vec2" not in ordered:
        ordered.append("wav2vec2")
    return ordered


# ── Per-sample processing ──────────────────────────────────────────────────────

def process_sample(
    idx: int,
    hyps: dict,
    ranked_models: list,
    client: Client,
    use_resolver: bool = True,
) -> dict:
    """
    Full pipeline for one sample.
    Segments → ROVER per segment → resolver → concatenate.
    """
    # segment all model outputs using Qwen as anchor
    segments = segment_hypotheses(hyps, anchor_model="qwen")

    sentence_transcripts = []
    sentence_results     = []

    for sent_idx, seg in enumerate(segments):
        # build ordered hypothesis list for ROVER
        ordered_hyps = [
            (model, seg[model])
            for model in ranked_models
            if model in seg and seg[model].strip()
        ]

        if not ordered_hyps:
            continue

        # ROVER on this segment (short sequence — clean alignment)
        rover_result = rover_combine(ordered_hyps)
        rover_text   = rover_result.transcript
        n_risky      = len(risky_disagreements(rover_result))

        # sentence-level semantic auditor
        llm_intervened = False
        if use_resolver:
            model_sentences = {model: seg.get(model, "") for model, _ in ordered_hyps}
            resolved_text, res_log = audit_sentence(
                rover_transcript=rover_text,
                model_sentences=model_sentences,
                word_confidences=rover_result.word_confidences,
                client=client,
            )
            llm_intervened = res_log.get("called_llm", False)
        else:
            resolved_text = rover_text
            res_log       = {"called_llm": False, "changed": False}

        # sentence-level confidence
        word_confs  = [c for _, c in rover_result.word_confidences]
        base_conf   = float(np.mean(word_confs)) if word_confs else 0.5
        penalty     = 0.05 * n_risky + (0.05 if llm_intervened else 0.0)
        sent_conf   = max(0.0, base_conf - penalty)

        sentence_transcripts.append(resolved_text)
        sentence_results.append({
            "sentence_idx":        sent_idx,
            "rover_transcript":    rover_text,
            "resolved_transcript": resolved_text,
            "sentence_confidence": sent_conf,
            "n_disagreements":     len(rover_result.disagreements),
            "n_risky":             n_risky,
            "llm_intervened":      llm_intervened,
        })

    # concatenate segment outputs directly — no reconstruction
    full_transcript   = " ".join(sentence_transcripts).strip()

    # dialect pass: phonetic scan + semantic judge on full transcript
    dialect_log = {"n_candidates": 0, "n_approved": 0, "substitutions": []}
    if use_resolver:
        full_transcript, dialect_log = run_dialect_pass(full_transcript, client)

    sentence_confs    = [s["sentence_confidence"] for s in sentence_results]
    utterance_conf    = float(np.mean(sentence_confs)) if sentence_confs else 0.0
    flagged_sentences = [
        s["sentence_idx"] for s in sentence_results
        if s["sentence_confidence"] < 0.7
    ]

    return {
        "dataset_index":        idx,
        "ref":                  hyps.get("ref", ""),
        "hyp":                  full_transcript,
        "sentence_results":     sentence_results,
        "dialect_log":          dialect_log,
        "utterance_confidence": utterance_conf,
        "flagged_sentences":    flagged_sentences,
        "n_sentences":          len(sentence_results),
    }


# ── Dataset runner ─────────────────────────────────────────────────────────────

def run_dataset(dataset, client, max_samples=None, rerun=False, no_resolver=False):
    print(f"\n── {dataset} | segment ROVER ──")

    model_samples = {}
    for model in ASR_MODELS:
        path = CANONICAL_FILES[(model, dataset)]
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    conf_data = {}
    for model in CONF_MODELS:
        conf_data[model] = load_confidence_by_index(model, dataset)

    indices = get_subset_indices(dataset)
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix      = "_no_resolver" if no_resolver else ""
    output_path = os.path.join(
        OUTPUT_DIR, f"rover_{dataset}_sub150{suffix}.json"
    )

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
        ref = model_samples["qwen"][idx]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({
                "dataset_index": idx, "ref": ref, "hyp": None,
                "qwen_verdict_p2": None, "sample_WER": None,
                "skipped": True,
            })
            continue

        hyps = {m: model_samples[m][idx]["hyp"] for m in ASR_MODELS}

        ranked = rank_models(conf_data, idx)

        try:
            result = process_sample(
                idx, hyps, ranked, client,
                use_resolver=not no_resolver,
            )
        except Exception as e:
            print(f"  ERROR sample {idx}: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "dataset_index": idx, "ref": ref, "hyp": None,
                "qwen_verdict_p2": None, "sample_WER": None,
                "error": True, "error_msg": str(e),
            })
            continue

        hyp            = result["hyp"]
        sample_wer_val = wer(normalise(ref), normalise(hyp))
        verdict        = ollama_mar(client, ref, hyp, sample_wer_val)
        time.sleep(0.05)

        results.append({
            "ref":                  ref,
            "hyp":                  hyp,
            "sample_WER":           sample_wer_val,
            "qwen_verdict_p2":      verdict,
            "utterance_confidence": result["utterance_confidence"],
            "flagged_sentences":    result["flagged_sentences"],
            "n_sentences":          result["n_sentences"],
            "dialect_n_candidates": result["dialect_log"]["n_candidates"],
            "dialect_n_approved":   result["dialect_log"]["n_approved"],
            "dataset_index":        idx,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results},
                          f, indent=2, ensure_ascii=False)
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results
             if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]

    corpus_wer_val = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid],
    ) if valid else None
    mar = (
        sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid)
        if valid else None
    )

    output = {
        "approach":                "rover",
        "dataset":                 dataset,
        "subset_indices":          indices,
        "corpus_wer":              corpus_wer_val,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "—"
    mar_str = f"{mar*100:.2f}%"            if mar is not None            else "—"
    print(f"  WER: {wer_str}  MAR: {mar_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resolver", action="store_true")
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    client = Client(host=OLLAMA_HOST)
    try:
        client.list()
        print("Ollama connected.")
        if not args.no_resolver and not check_auditor_available(client):
            print("WARNING: resolver model not available — running without resolver")
            args.no_resolver = True
    except Exception as e:
        print(f"ERROR: could not connect to Ollama\n{e}")
        return

    run_dataset(
        args.dataset, client,
        max_samples=args.max_samples,
        rerun=args.rerun,
        no_resolver=args.no_resolver,
    )


if __name__ == "__main__":
    main()