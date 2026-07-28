"""
rerunning/sentence_confidence/compute_acoustic_confidence.py

Acoustic confidence (Method 3). --variant selects which naive combo
file to anchor on (confscore or probscore), same as
compute_crossmodel_agreement.py, for a fully symmetric comparison.

Usage:
    python rerunning/sentence_confidence/compute_acoustic_confidence.py --dataset commonvoice --variant probscore
    python rerunning/sentence_confidence/compute_acoustic_confidence.py --dataset commonvoice --variant confscore
"""

import json
import osen
import string
import difflib
import argparse
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


def find_best_span_indices(anchor, word_segments, search_start=0, max_extra=10):
    anchor_tokens = tokenise(anchor)
    if not anchor_tokens:
        return None, None
    seg_tokens = [normalise_word(w.get("word", "")) for w in word_segments]
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
    return best_start, best_end


def mean_conf_for_span(word_segments, start, end):
    if start is None or end is None:
        return 0.5
    confs = [w.get("confidence") for w in word_segments[start:end] if w.get("confidence") is not None]
    return sum(confs) / len(confs) if confs else 0.5


def wx_conf_by_time(word_segments, sent_start, sent_end):
    words = [w for w in word_segments if w.get("start") is not None and w.get("end") is not None
             and w["start"] >= sent_start - 0.1 and w["end"] <= sent_end + 0.1]
    if not words:
        return None
    return sum(w.get("confidence", 0.8) for w in words) / len(words)


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
    print(f"\n-- Acoustic confidence: {dataset} ({variant}) --")

    output_path = os.path.join(OUTPUT_DIR, f"acoustic_confidence_{variant}_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print("  Already exists. Use --rerun to recompute.")
        return

    combo_samples, corpus_wer = load_combo(dataset, variant)
    segs_by_model = {m: load_model_samples(m, dataset) for m in ASR_MODELS}
    if not combo_samples:
        return

    output_samples = []
    all_mean = []
    all_by_model = {m: [] for m in ASR_MODELS}

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
        wx_segs = model_segs.get("whisperx", [])
        total_dur = max((wx_segs[-1].get("end") or 0) if wx_segs else 0, 0.1)
        n_sent = len(sent_confs)

        sentences = []
        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue
            frac = sent_pos / max(n_sent, 1)
            est_start, est_end = frac * total_dur, (frac + 1 / max(n_sent, 1)) * total_dur

            model_confs = {}
            wx_conf = wx_conf_by_time(wx_segs, est_start, est_end)
            if wx_conf is None:
                s_idx, e_idx = find_best_span_indices(anchor, wx_segs, max(0, int(frac * n_segs["whisperx"]) - 5))
                wx_conf = mean_conf_for_span(wx_segs, s_idx, e_idx)
            model_confs["whisperx"] = wx_conf

            for m in ["qwen", "parakeet", "wav2vec2"]:
                search = max(0, int(frac * n_segs[m]) - 5)
                s_idx, e_idx = find_best_span_indices(anchor, model_segs[m], max(cursors[m], search))
                model_confs[m] = mean_conf_for_span(model_segs[m], s_idx, e_idx)
                cursors[m] = e_idx or cursors[m]

            mean_conf = round(sum(model_confs.values()) / len(model_confs), 4)
            all_mean.append(mean_conf)
            for m in ASR_MODELS:
                all_by_model[m].append(model_confs[m])

            sentences.append({"sent_pos": sent_pos, "confidence": mean_conf,
                               **{f"{m}_acoustic": round(model_confs[m], 4) for m in ASR_MODELS},
                               "sentence": anchor})

        output_samples.append({"dataset_index": dataset_index, "sentences": sentences})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"method": "acoustic_mean", "dataset": dataset, "variant": variant,
                    "corpus_wer": corpus_wer, "samples": output_samples}, f, indent=2, ensure_ascii=False)

    n = len(all_mean)
    if n:
        print(f"  Total sentences: {n}")
        print(f"  Mean acoustic (mean): {sum(all_mean)/n:.3f}")
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