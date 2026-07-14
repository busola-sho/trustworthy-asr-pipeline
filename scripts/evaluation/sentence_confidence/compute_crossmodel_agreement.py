"""
scripts/evaluation/sentence_confidence/compute_crossmodel_agreement.py

Method 2: Cross-model agreement confidence.

For each pipeline sentence (used as position anchor only):
  1. Find matching span in WhisperX word segments
  2. Find matching span in Qwen word segments
  3. Find matching span in Parakeet word segments
  4. Compute PAIRWISE similarity between the three model spans:
       sim(wx, qwen), sim(wx, parakeet), sim(qwen, parakeet)
  5. mean_agreement = mean of 3 pairwise similarities
  6. min_agreement  = min of 3 pairwise similarities

The selector sentence is used only as a position anchor — it does NOT
participate in the similarity calculation. Confidence is purely
model-vs-model agreement, independent of the selector.

This also reflects deployment: selector outputs final sentences,
pairwise agreement scores their confidence without needing a reference.

Output: results/sentence_confidence/crossmodel_agreement_{dataset}.json

Usage:
    python scripts/evaluation/sentence_confidence/compute_crossmodel_agreement.py --dataset commonvoice
    python scripts/evaluation/sentence_confidence/compute_crossmodel_agreement.py --dataset all
"""

import json
import os
import string
import difflib
import argparse
from typing import List, Dict, Optional

OUTPUT_DIR = "results/sentence_confidence"

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
    ("qwen",     "commonvoice"):      "results/benchmarks/subsets/qwen3asr_commonvoice_sub150.json",
    ("qwen",     "edacc"):            "results/benchmarks/subsets/qwen3asr_edacc_sub150.json",
    ("qwen",     "english_dialects"): "results/benchmarks/subsets/qwen3asr_english_dialects_sub150.json",
    ("parakeet", "commonvoice"):      "results/benchmarks/subsets/parakeet_commonvoice_sub150.json",
    ("parakeet", "edacc"):            "results/benchmarks/subsets/parakeet_edacc_sub150.json",
    ("parakeet", "english_dialects"): "results/benchmarks/subsets/parakeet_english_dialects_sub150.json",
    ("whisperx", "shetland"):          "results/benchmarks/subsets/shetland/whisperx_shetland_sub100.json",
    ("qwen",     "shetland"):          "results/benchmarks/subsets/shetland/qwen3asr_shetland_sub100.json",
    ("parakeet", "shetland"):          "results/benchmarks/subsets/shetland/parakeet_shetland_sub100.json",
}

CORPUS_WER = {
    "commonvoice":      0.18328,
    "edacc":            0.19747,
    "english_dialects": 0.03809,
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalise_word(word: str) -> str:
    return word.lower().strip().strip(string.punctuation)


def tokenise(text: str) -> List[str]:
    return [normalise_word(w) for w in text.split() if normalise_word(w)]


def pairwise_similarity(text_a: str, text_b: str) -> float:
    """Token-level SequenceMatcher similarity between two text spans."""
    tokens_a = tokenise(text_a)
    tokens_b = tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    return difflib.SequenceMatcher(None, tokens_a, tokens_b).ratio()


# ── Span extraction ────────────────────────────────────────────────────────────

def find_best_span(
    anchor_sentence: str,
    word_segments:   List[Dict],
    search_start:    int = 0,
    max_extra:       int = 10,
) -> str:
    """
    Find the best matching span of words in word_segments for an anchor sentence.
    The anchor is used only for position — it is NOT compared against the span.
    Returns the matched span text.
    """
    anchor_tokens = tokenise(anchor_sentence)
    if not anchor_tokens:
        return ""

    all_words  = [w.get("word", "") for w in word_segments]
    seg_tokens = [normalise_word(w) for w in all_words if normalise_word(w)]

    if not seg_tokens:
        return ""

    sent_len  = len(anchor_tokens)
    min_len   = max(1, sent_len - max_extra)
    max_len   = sent_len + max_extra
    search_end = min(len(seg_tokens), search_start + sent_len * 3 + max_extra)

    best_start, best_end, best_score = None, None, 0.0

    for start in range(search_start, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(seg_tokens):
                continue
            candidate = seg_tokens[start:end]
            score = difflib.SequenceMatcher(None, anchor_tokens, candidate).ratio()
            if score > best_score:
                best_score = score
                best_start = start
                best_end   = end
        if best_score >= 0.88 and start > search_start + sent_len + 5:
            break

    if best_start is None:
        return ""

    # reconstruct span from original (unnormalised) words
    # need to map back through non-empty words
    non_empty_indices = [i for i, w in enumerate(all_words) if normalise_word(w)]
    if best_end > len(non_empty_indices):
        return ""

    orig_start = non_empty_indices[best_start]
    orig_end   = non_empty_indices[best_end - 1] + 1
    return " ".join(all_words[orig_start:orig_end])


# ── Data loading ───────────────────────────────────────────────────────────────

def load_subset_by_index(model: str, dataset: str) -> Dict[int, Dict]:
    path = SUBSET_FILES.get((model, dataset))
    if not path or not os.path.exists(path):
        print(f"  WARNING: subset file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        s["sample_index"]: s
        for s in data.get("samples", [])
        if s.get("sample_index") is not None
    }


def load_combo(dataset: str) -> List[Dict]:
    path = COMBO_FILES.get(dataset)
    if not path or not os.path.exists(path):
        print(f"  ERROR: combo file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", [])


# ── Main computation ───────────────────────────────────────────────────────────

def compute_dataset(dataset: str, rerun: bool = False):
    print(f"\n── Cross-model agreement: {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"crossmodel_agreement_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print(f"  Already exists. Use --rerun to recompute.")
        return

    combo_samples = load_combo(dataset)
    wx_by_idx     = load_subset_by_index("whisperx", dataset)
    qwen_by_idx   = load_subset_by_index("qwen",     dataset)
    par_by_idx    = load_subset_by_index("parakeet", dataset)

    if not combo_samples:
        return

    output_samples = []
    all_mean_agreements = []
    all_min_agreements  = []

    for s in combo_samples:
        if s.get("skipped") or s.get("error"):
            continue

        dataset_index = s.get("dataset_index")
        sent_confs    = s.get("sentence_confidences", [])

        if dataset_index is None or not sent_confs:
            continue

        wx_segs  = wx_by_idx.get(dataset_index,  {}).get("segments", [])
        qw_segs  = qwen_by_idx.get(dataset_index, {}).get("segments", [])
        par_segs = par_by_idx.get(dataset_index,  {}).get("segments", [])

        n_sent = len(sent_confs)
        n_wx   = len(wx_segs)
        n_qw   = len(qw_segs)
        n_par  = len(par_segs)

        sentences  = []
        wx_cursor  = 0
        qw_cursor  = 0
        par_cursor = 0

        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue

            # proportional search start
            frac = sent_pos / max(n_sent, 1)
            wx_search  = max(0, int(frac * n_wx)  - 5)
            qw_search  = max(0, int(frac * n_qw)  - 5)
            par_search = max(0, int(frac * n_par) - 5)

            # find best span in each model using anchor as position guide
            wx_span  = find_best_span(anchor, wx_segs,  search_start=max(wx_cursor,  wx_search))
            qw_span  = find_best_span(anchor, qw_segs,  search_start=max(qw_cursor,  qw_search))
            par_span = find_best_span(anchor, par_segs, search_start=max(par_cursor, par_search))

            # fallback to anchor if span not found
            wx_span  = wx_span  or anchor
            qw_span  = qw_span  or anchor
            par_span = par_span or anchor

            # PAIRWISE similarity between model spans
            # anchor sentence does NOT participate in similarity
            sim_wx_qw   = pairwise_similarity(wx_span,  qw_span)
            sim_wx_par  = pairwise_similarity(wx_span,  par_span)
            sim_qw_par  = pairwise_similarity(qw_span,  par_span)

            mean_agreement = round((sim_wx_qw + sim_wx_par + sim_qw_par) / 3, 4)
            min_agreement  = round(min(sim_wx_qw, sim_wx_par, sim_qw_par), 4)

            all_mean_agreements.append(mean_agreement)
            all_min_agreements.append(min_agreement)

            sentences.append({
                "sent_pos":       sent_pos,
                "confidence":     mean_agreement,
                "min_agreement":  min_agreement,
                "sim_wx_qwen":    round(sim_wx_qw,  4),
                "sim_wx_par":     round(sim_wx_par, 4),
                "sim_qwen_par":   round(sim_qw_par, 4),
                "sentence":       anchor,
            })

            # advance cursors
            wx_cursor  = wx_search
            qw_cursor  = qw_search
            par_cursor = par_search

        output_samples.append({
            "dataset_index": dataset_index,
            "sentences":     sentences,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "method":     "crossmodel_agreement",
        "dataset":    dataset,
        "corpus_wer": CORPUS_WER.get(dataset),
        "samples":    output_samples,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n = len(all_mean_agreements)
    print(f"  Total sentences: {n}")
    print(f"  Mean pairwise agreement (mean): {sum(all_mean_agreements)/n:.3f}")
    print(f"  Mean pairwise agreement (min):  {sum(all_min_agreements)/n:.3f}")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-model pairwise agreement confidence scores"
    )
    parser.add_argument("--dataset", default="all",
                        choices=DATASETS + ["all"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        compute_dataset(dataset, rerun=args.rerun)


if __name__ == "__main__":
    main()