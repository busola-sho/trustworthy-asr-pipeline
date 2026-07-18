"""
scripts/evaluation/sentence_confidence/run_sentence_severity_per_model.py

For each selector sentence (position anchor), finds the matching span in
each individual model's transcript and runs the sentence severity judge.

This gives sentence-level severity scores per model for comparison against
the ensemble, using consistent sentence boundaries across all models.

Output: results/sentence_confidence/sentence_severity_{model}_{dataset}.json
  {
    "model": str,
    "dataset": str,
    "rows": [
      {
        "dataset_index": int,
        "sent_pos": int,
        "selector_sentence": str,   ← anchor (from ensemble)
        "model_span": str,          ← matched span from model output
        "severity": int,            ← 0-4
      }
    ]
  }

Usage:
    python scripts/evaluation/sentence_confidence/run_sentence_severity_per_model.py --dataset commonvoice
    python scripts/evaluation/sentence_confidence/run_sentence_severity_per_model.py --dataset all --model all
"""

import json
import os
import argparse
import string
import difflib
import time
from collections import defaultdict
from typing import List, Dict, Optional
from ollama import Client
from src.judge import ollama_sentence_severity

OLLAMA_HOST = "http://localhost:11434"
OUTPUT_DIR  = "results/sentence_confidence"

COMBO_FILES = {
    "commonvoice":      "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json",
    "edacc":            "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json",
    "english_dialects": "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_english_dialects_qwen_sub150.json",
    "shetland":         "results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_shetland_qwen_sub150.json",
}

SUBSET_FILES = {
    ("whisperx", "commonvoice"):      "results/benchmarks/subsets/whisperx_commonvoice_sub150.json",
    ("whisperx", "edacc"):            "results/benchmarks/subsets/whisperx_edacc_sub150.json",
    ("whisperx", "english_dialects"): "results/benchmarks/subsets/whisperx_english_dialects_sub150.json",
    ("whisperx", "shetland"):         "results/benchmarks/subsets/shetland/whisperx_shetland_sub100.json",
    ("qwen",     "commonvoice"):      "results/benchmarks/subsets/qwen3asr_commonvoice_sub150.json",
    ("qwen",     "edacc"):            "results/benchmarks/subsets/qwen3asr_edacc_sub150.json",
    ("qwen",     "english_dialects"): "results/benchmarks/subsets/qwen3asr_english_dialects_sub150.json",
    ("qwen",     "shetland"):         "results/benchmarks/subsets/shetland/qwen3asr_shetland_sub100.json",
    ("parakeet", "commonvoice"):      "results/benchmarks/subsets/parakeet_commonvoice_sub150.json",
    ("parakeet", "edacc"):            "results/benchmarks/subsets/parakeet_edacc_sub150.json",
    ("parakeet", "english_dialects"): "results/benchmarks/subsets/parakeet_english_dialects_sub150.json",
    ("parakeet", "shetland"):         "results/benchmarks/subsets/shetland/parakeet_shetland_sub100.json",
}

MODELS   = ["whisperx", "qwen", "parakeet"]
DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalise_word(word: str) -> str:
    return word.lower().strip().strip(string.punctuation)


def find_best_span(anchor: str, words: List[str], search_start: int = 0, max_extra: int = 10) -> str:
    """Find best matching span of words for anchor sentence. Returns span text."""
    anchor_tokens = [normalise_word(w) for w in anchor.split() if normalise_word(w)]
    word_tokens   = [normalise_word(w) for w in words]

    if not anchor_tokens or not word_tokens:
        return anchor

    sent_len   = len(anchor_tokens)
    min_len    = max(1, sent_len - max_extra)
    max_len    = sent_len + max_extra
    search_end = min(len(word_tokens), search_start + sent_len * 3 + max_extra)

    best_start, best_end, best_score = None, None, 0.0

    for start in range(search_start, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(word_tokens):
                continue
            score = difflib.SequenceMatcher(None, anchor_tokens, word_tokens[start:end]).ratio()
            if score > best_score:
                best_score = score
                best_start = start
                best_end   = end
        if best_score >= 0.88 and start > search_start + sent_len + 5:
            break

    if best_start is None:
        return anchor  # fallback to anchor

    return " ".join(words[best_start:best_end])


# ── Main ───────────────────────────────────────────────────────────────────────

def run_model_dataset(model: str, dataset: str, client: Client, rerun: bool = False):
    print(f"\n── {model} / {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"sentence_severity_{model}_{dataset}.json")

    # load existing rows for resuming
    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing_data = json.load(f)
        rows = existing_data.get("rows", [])
        done = {(r["dataset_index"], r["sent_pos"]) for r in rows}
        print(f"  Resuming — {len(rows)} rows already done")
    else:
        rows = []
        done = set()

    # load combo file (selector output — sentence boundaries + ref)
    combo_path = COMBO_FILES.get(dataset)
    if not combo_path or not os.path.exists(combo_path):
        print(f"  ERROR: combo file not found: {combo_path}")
        return
    with open(combo_path) as f:
        combo = json.load(f)

    # load individual model subset file (for hyp words)
    subset_path = SUBSET_FILES.get((model, dataset))
    if not subset_path or not os.path.exists(subset_path):
        print(f"  ERROR: subset file not found: {subset_path}")
        return
    with open(subset_path) as f:
        subset = json.load(f)

    # build lookup: sample_index -> hyp text
    hyp_by_index = {
        s["sample_index"]: s.get("hyp", "")
        for s in subset.get("samples", [])
        if s.get("sample_index") is not None
    }

    new_rows = 0
    start_time = time.time()

    for s in combo.get("samples", []):
        if s.get("skipped") or s.get("error"):
            continue

        dataset_index = s.get("dataset_index")
        ref           = s.get("ref", "")
        sent_confs    = s.get("sentence_confidences", [])

        if dataset_index is None or not ref or not sent_confs:
            continue

        # get model's full hypothesis
        model_hyp = hyp_by_index.get(dataset_index, "")
        if not model_hyp:
            continue

        model_words  = model_hyp.split()
        n_sent       = len(sent_confs)
        search_cursor = 0

        for sent_pos, sc in enumerate(sent_confs):
            key = (dataset_index, sent_pos)
            if key in done:
                continue

            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue

            # estimate search start proportionally
            frac         = sent_pos / max(n_sent, 1)
            search_start = max(0, int(frac * len(model_words)) - 5)

            # find matching span in model output
            model_span = find_best_span(
                anchor, model_words,
                search_start=max(search_cursor, search_start)
            )

            # run severity judge
            severity = ollama_sentence_severity(client, ref, model_span)

            rows.append({
                "dataset_index":    dataset_index,
                "sent_pos":         sent_pos,
                "selector_sentence": anchor,
                "model_span":       model_span,
                "severity":         severity,
            })
            done.add(key)
            new_rows += 1

            # advance cursor
            n_anchor_words = len(anchor.split())
            search_cursor  = max(search_cursor, search_start + n_anchor_words)

        # save every 10 samples
        if new_rows > 0 and new_rows % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {len(rows)} rows total ({elapsed:.0f}s)")
            _save(output_path, model, dataset, rows)

    _save(output_path, model, dataset, rows)

    valid = [r for r in rows if r.get("severity") is not None]
    from collections import Counter
    dist = Counter(r["severity"] for r in valid)
    print(f"  Total: {len(valid)} | Severity dist: {dict(sorted(dist.items()))}")


def _save(output_path, model, dataset, rows):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model":   model,
            "dataset": dataset,
            "rows":    rows,
        }, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run sentence severity judge on individual model outputs"
    )
    parser.add_argument("--model",   default="all", choices=MODELS + ["all"])
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--rerun",   action="store_true")
    args = parser.parse_args()

    client   = Client(host=OLLAMA_HOST)
    models   = MODELS   if args.model   == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    for model in models:
        for dataset in datasets:
            run_model_dataset(model, dataset, client, rerun=args.rerun)

    print("\nDone. Now plot with:")
    print("  python scripts/evaluation/sentence_confidence/plot_severity_distribution.py")


if __name__ == "__main__":
    main()