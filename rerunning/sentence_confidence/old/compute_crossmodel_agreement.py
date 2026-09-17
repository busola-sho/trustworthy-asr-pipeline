"""
rerunning/sentence_confidence/compute_crossmodel_agreement.py

Cross-model agreement (Method 2). --variant selects which naive
combo file to anchor sentence boundaries on (confscore or probscore) -
each produces its own output file, so both can be computed
independently for a fully symmetric comparison.

Usage:
    python rerunning/sentence_confidence/compute_crossmodel_agreement.py --dataset commonvoice --variant probscore
    python rerunning/sentence_confidence/compute_crossmodel_agreement.py --dataset commonvoice --variant confscore
"""

import json
import os
import string
import difflib
import argparse
import itertools
from typing import List, Dict

from src.selector import find_canonical_file, load_samples

OUTPUT_DIR = "results/sentence_confidence"
ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS   = ["commonvoice", "edacc", "english_dialects", "shetland"]


def combo_path_for(dataset, variant):
    split = "full" if dataset == "shetland" else "dev"
    return f"writeup_results/ensembles/naive_{variant}/naive_{variant}_{dataset}_gemma4sel_{split}.json"


def normalise_word(w): return w.lower().strip().strip(string.punctuation)
def tokenise(text): return [normalise_word(w) for w in text.split() if normalise_word(w)]


def pairwise_similarity(a, b):
    ta, tb = tokenise(a), tokenise(b)
    if not ta or not tb:
        return 0.0
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def find_best_span(anchor, word_segments, search_start=0, max_extra=10):
    anchor_tokens = tokenise(anchor)
    if not anchor_tokens:
        return ""
    all_words = [w.get("word", "") for w in word_segments]
    seg_tokens = [normalise_word(w) for w in all_words if normalise_word(w)]
    if not seg_tokens:
        return ""
    sent_len = len(anchor_tokens)
    min_len, max_len = max(1, sent_len - max_extra), sent_len + max_extra
    search_end = min(len(seg_tokens), search_start + sent_len * 3 + max_extra)
    best_start, best_end, best_score = None, None, 0.0
    for start in range(search_start, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(seg_tokens):
                continue
            score = difflib.SequenceMatcher(None, anchor_tokens, seg_tokens[start:end]).ratio()
            if score > best_score:
                best_score, best_start, best_end = score, start, end
        if best_score >= 0.88 and start > search_start + sent_len + 5:
            break
    if best_start is None:
        return ""
    non_empty = [i for i, w in enumerate(all_words) if normalise_word(w)]
    if best_end > len(non_empty):
        return ""
    return " ".join(all_words[non_empty[best_start]:non_empty[best_end - 1] + 1])


def load_model_samples(model, dataset):
    try:
        path = find_canonical_file(model, dataset)
    except FileNotFoundError:
        print(f"  WARNING: no canonical file for {model}/{dataset}")
        return {}
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def load_combo(dataset, variant):
    path = combo_path_for(dataset, variant)
    if not os.path.exists(path):
        print(f"  ERROR: combo file not found: {path}")
        return [], None
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", []), data.get("corpus_wer")


def compute_dataset(dataset, variant, rerun=False):
    print(f"\n-- Cross-model agreement: {dataset} ({variant}) --")

    output_path = os.path.join(OUTPUT_DIR, f"crossmodel_agreement_{variant}_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print("  Already exists. Use --rerun to recompute.")
        return

    combo_samples, corpus_wer = load_combo(dataset, variant)
    segs_by_model = {m: load_model_samples(m, dataset) for m in ASR_MODELS}
    if not combo_samples:
        return

    model_pairs = list(itertools.combinations(ASR_MODELS, 2))
    output_samples = []
    all_mean, all_min = [], []

    for s in combo_samples:
        if s.get("skipped") or s.get("error"):
            continue
        dataset_index = s.get("dataset_index")
        sent_confs = s.get("sentence_confidences", [])
        if dataset_index is None or not sent_confs:
            continue

        model_segs = {m: segs_by_model[m].get(dataset_index, {}).get("segments", []) for m in ASR_MODELS}
        n_segs = {m: len(model_segs[m]) for m in ASR_MODELS}
        cursors = {m: 0 for m in ASR_MODELS}
        n_sent = len(sent_confs)

        sentences = []
        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue
            frac = sent_pos / max(n_sent, 1)
            spans = {}
            for m in ASR_MODELS:
                search = max(0, int(frac * n_segs[m]) - 5)
                span = find_best_span(anchor, model_segs[m], max(cursors[m], search))
                spans[m] = span or anchor
                cursors[m] = search

            pair_sims = {f"sim_{a}_{b}": round(pairwise_similarity(spans[a], spans[b]), 4) for a, b in model_pairs}
            sims = list(pair_sims.values())
            mean_agreement = round(sum(sims) / len(sims), 4)
            min_agreement = round(min(sims), 4)
            all_mean.append(mean_agreement)
            all_min.append(min_agreement)

            sentences.append({"sent_pos": sent_pos, "confidence": mean_agreement,
                               "min_agreement": min_agreement, **pair_sims, "sentence": anchor})

        output_samples.append({"dataset_index": dataset_index, "sentences": sentences})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"method": "crossmodel_agreement", "dataset": dataset, "variant": variant,
                    "corpus_wer": corpus_wer, "models": ASR_MODELS, "samples": output_samples},
                   f, indent=2, ensure_ascii=False)

    n = len(all_mean)
    if n:
        print(f"  Total sentences: {n}")
        print(f"  Mean agreement (mean): {sum(all_mean)/n:.3f}")
        print(f"  Mean agreement (min):  {sum(all_min)/n:.3f}")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--variant", required=True, choices=["confscore", "probscore"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for d in datasets:
        compute_dataset(d, args.variant, rerun=args.rerun)


if __name__ == "__main__":
    main()