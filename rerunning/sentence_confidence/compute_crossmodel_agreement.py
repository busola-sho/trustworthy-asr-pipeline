"""
rerunning/sentence_confidence/compute_crossmodel_agreement.py

Method 2: Cross-model agreement confidence.

UPDATED to work with naive (your locked best technique) instead of the
old context_v2 + qwen2.5 pipeline:
  - COMBO_FILES now points at naive_probscore's output (any of the 4
    naive_*score* variants works equally well here - they all produce
    the SAME combined transcript/sentence breakdown at temperature=0,
    since only the confidence-request portion of the prompt differs
    between variants; the actual "sentence" anchors used by this script
    are identical regardless of which variant you point it at)
  - Uses your CURRENT canonical benchmark files (via find_canonical_file)
    instead of the old static 150-sample subset paths
  - Extended from 3 models (whisperx/qwen/parakeet) to all 4
    (+ wav2vec2), since naive actually uses all 4 and all 4 now have
    real confidence data - agreement is now the mean/min of 6 pairwise
    similarities instead of 3
  - corpus_wer is read directly from the combo file's own field instead
    of a hardcoded dict of stale numbers

For each pipeline sentence (used as position anchor only):
  1. Find matching span in each of the 4 models' word segments
  2. Compute PAIRWISE similarity between all C(4,2)=6 model-pair spans
  3. mean_agreement = mean of the 6 pairwise similarities
  4. min_agreement  = min of the 6 pairwise similarities

The selector sentence is used only as a position anchor - it does NOT
participate in the similarity calculation. Confidence is purely
model-vs-model agreement, independent of the selector.

Output: results/sentence_confidence/crossmodel_agreement_{dataset}.json

Usage:
    python rerunning/sentence_confidence/compute_crossmodel_agreement.py --dataset commonvoice
    python rerunning/sentence_confidence/compute_crossmodel_agreement.py --dataset all
"""

import json
import os
import string
import difflib
import argparse
import itertools
from typing import List, Dict

from src.selector import find_canonical_file, load_samples

OUTPUT_DIR = "writeup_results/sentence_confidence"

# any one of the 4 naive_*score* variants works - they share the same
# combined transcript/sentence breakdown at temperature=0
COMBO_FILES = {
    "commonvoice":      "writeup_results/ensembles/naive_probscore/naive_probscore_commonvoice_gemma4sel_dev.json",
    "edacc":            "writeup_results/ensembles/naive_probscore/naive_probscore_edacc_gemma4sel_dev.json",
    "english_dialects": "writeup_results/ensembles/naive_probscore/naive_probscore_english_dialects_gemma4sel_dev.json",
    "shetland":         "writeup_results/ensembles/naive_probscore/naive_probscore_shetland_gemma4sel_full.json",
}

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS   = ["commonvoice", "edacc", "english_dialects", "shetland"]


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
    The anchor is used only for position - it is NOT compared against the span.
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

    non_empty_indices = [i for i, w in enumerate(all_words) if normalise_word(w)]
    if best_end > len(non_empty_indices):
        return ""

    orig_start = non_empty_indices[best_start]
    orig_end   = non_empty_indices[best_end - 1] + 1
    return " ".join(all_words[orig_start:orig_end])


# ── Data loading ───────────────────────────────────────────────────────────────

def load_model_samples(model: str, dataset: str) -> Dict[int, Dict]:
    """Loads a model's benchmark samples indexed by sample_index, using
    your CURRENT canonical file resolution (not the old static
    150-sample subset paths)."""
    try:
        path = find_canonical_file(model, dataset)
    except FileNotFoundError:
        print(f"  WARNING: no canonical file found for {model}/{dataset}")
        return {}
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def load_combo(dataset: str) -> tuple:
    """Returns (samples, corpus_wer) from the naive combo file - corpus_wer
    read directly from the file's own field, not a hardcoded dict."""
    path = COMBO_FILES.get(dataset)
    if not path or not os.path.exists(path):
        print(f"  ERROR: combo file not found: {path}")
        return [], None
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", []), data.get("corpus_wer")


# ── Main computation ───────────────────────────────────────────────────────────

def compute_dataset(dataset: str, rerun: bool = False):
    print(f"\n── Cross-model agreement: {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"crossmodel_agreement_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print(f"  Already exists. Use --rerun to recompute.")
        return

    combo_samples, corpus_wer = load_combo(dataset)
    segs_by_model = {m: load_model_samples(m, dataset) for m in ASR_MODELS}

    if not combo_samples:
        return

    model_pairs = list(itertools.combinations(ASR_MODELS, 2))  # 6 pairs for 4 models

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

        model_segs = {
            m: segs_by_model[m].get(dataset_index, {}).get("segments", [])
            for m in ASR_MODELS
        }
        n_segs = {m: len(model_segs[m]) for m in ASR_MODELS}

        sentences = []
        cursors   = {m: 0 for m in ASR_MODELS}
        n_sent    = len(sent_confs)

        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue

            frac = sent_pos / max(n_sent, 1)
            spans = {}
            for m in ASR_MODELS:
                search = max(0, int(frac * n_segs[m]) - 5)
                span = find_best_span(anchor, model_segs[m], search_start=max(cursors[m], search))
                spans[m] = span or anchor
                cursors[m] = search

            pair_sims = {
                f"sim_{a}_{b}": round(pairwise_similarity(spans[a], spans[b]), 4)
                for a, b in model_pairs
            }
            sims = list(pair_sims.values())
            mean_agreement = round(sum(sims) / len(sims), 4)
            min_agreement  = round(min(sims), 4)

            all_mean_agreements.append(mean_agreement)
            all_min_agreements.append(min_agreement)

            sentences.append({
                "sent_pos":      sent_pos,
                "confidence":    mean_agreement,
                "min_agreement": min_agreement,
                **pair_sims,
                "sentence":      anchor,
            })

        output_samples.append({
            "dataset_index": dataset_index,
            "sentences":     sentences,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = {
        "method":     "crossmodel_agreement",
        "dataset":    dataset,
        "corpus_wer": corpus_wer,
        "models":     ASR_MODELS,
        "samples":    output_samples,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n = len(all_mean_agreements)
    if n:
        print(f"  Total sentences: {n}")
        print(f"  Mean pairwise agreement (mean): {sum(all_mean_agreements)/n:.3f}")
        print(f"  Mean pairwise agreement (min):  {sum(all_min_agreements)/n:.3f}")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-model pairwise agreement confidence scores"
    )
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        compute_dataset(dataset, rerun=args.rerun)


if __name__ == "__main__":
    main()