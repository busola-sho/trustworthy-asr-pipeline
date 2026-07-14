"""
scripts/evaluation/sentence_confidence/compute_acoustic_confidence.py

Method 3: Acoustic confidence.

For each pipeline sentence:
  1. WhisperX: words in sentence time range (timestamp-based) -> mean confidence
  2. Qwen: best matching span via constrained text search -> mean confidence
  3. Parakeet: best matching span via constrained text search -> mean confidence
  4. mean_acoustic = mean(whisperx, qwen, parakeet)

Also saves individual model variants:
  acoustic_whisperx_{dataset}.json
  acoustic_qwen_{dataset}.json
  acoustic_parakeet_{dataset}.json

Usage:
    python scripts/evaluation/sentence_confidence/compute_acoustic_confidence.py --dataset commonvoice
    python scripts/evaluation/sentence_confidence/compute_acoustic_confidence.py --dataset all
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

CORPUS_WER = {
    "commonvoice":      0.18328,
    "edacc":            0.19747,
    "english_dialects": 0.03809,
    "shetland":         0.1150,
}

DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


def normalise_word(word: str) -> str:
    return word.lower().strip().strip(string.punctuation)


def tokenise(text: str) -> List[str]:
    return [normalise_word(w) for w in text.split() if normalise_word(w)]


def find_best_span_indices(anchor, word_segments, search_start=0, max_extra=10):
    anchor_tokens = tokenise(anchor)
    if not anchor_tokens:
        return None, None
    seg_tokens = [normalise_word(w.get("word", "")) for w in word_segments]
    sent_len   = len(anchor_tokens)
    min_len    = max(1, sent_len - max_extra)
    max_len    = sent_len + max_extra
    search_end = min(len(seg_tokens), search_start + sent_len * 3 + max_extra)
    best_start, best_end, best_score = None, None, 0.0
    for start in range(search_start, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(seg_tokens):
                continue
            score = difflib.SequenceMatcher(None, anchor_tokens, seg_tokens[start:end]).ratio()
            if score > best_score:
                best_score = score
                best_start = start
                best_end   = end
        if best_score >= 0.88 and start > search_start + sent_len + 5:
            break
    return best_start, best_end


def mean_conf_for_span(word_segments, start, end):
    if start is None or end is None:
        return 0.5
    confs = [w.get("confidence") for w in word_segments[start:end] if w.get("confidence") is not None]
    return sum(confs) / len(confs) if confs else 0.5


def wx_conf_by_time(word_segments, sent_start, sent_end):
    words = [
        w for w in word_segments
        if w.get("start") is not None and w.get("end") is not None
        and w["start"] >= sent_start - 0.1
        and w["end"]   <= sent_end   + 0.1
    ]
    if not words:
        return None
    confs = [w.get("confidence", 0.8) for w in words]
    return sum(confs) / len(confs)


def load_subset_by_index(model, dataset):
    path = SUBSET_FILES.get((model, dataset))
    if not path or not os.path.exists(path):
        print(f"  WARNING: not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {s["sample_index"]: s for s in data.get("samples", []) if s.get("sample_index") is not None}


def load_combo(dataset):
    path = COMBO_FILES.get(dataset)
    if not path or not os.path.exists(path):
        print(f"  ERROR: combo file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", [])


def _save(path, method, dataset, corpus_wer, samples):
    with open(path, "w") as f:
        json.dump({"method": method, "dataset": dataset,
                   "corpus_wer": corpus_wer, "samples": samples},
                  f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def compute_dataset(dataset, rerun=False):
    print(f"\n-- Acoustic confidence: {dataset} --")

    output_path = os.path.join(OUTPUT_DIR, f"acoustic_confidence_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print(f"  Already exists. Use --rerun to recompute.")
        return

    combo_samples = load_combo(dataset)
    wx_by_idx     = load_subset_by_index("whisperx", dataset)
    qw_by_idx     = load_subset_by_index("qwen",     dataset)
    par_by_idx    = load_subset_by_index("parakeet", dataset)

    if not combo_samples:
        return

    output_samples = []
    all_mean, all_wx, all_qw, all_par = [], [], [], []

    for s in combo_samples:
        if s.get("skipped") or s.get("error"):
            continue

        dataset_index = s.get("dataset_index")
        sent_confs    = s.get("sentence_confidences", [])
        if dataset_index is None or not sent_confs:
            continue

        wx_segs  = wx_by_idx.get(dataset_index,  {}).get("segments", [])
        qw_segs  = qw_by_idx.get(dataset_index,  {}).get("segments", [])
        par_segs = par_by_idx.get(dataset_index, {}).get("segments", [])

        n_sent = len(sent_confs)
        n_qw   = len(qw_segs)
        n_par  = len(par_segs)

        sentences  = []
        qw_cursor  = 0
        par_cursor = 0

        total_dur = max((wx_segs[-1].get("end") or 0) if wx_segs else 0, 0.1)

        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue

            frac      = sent_pos / max(n_sent, 1)
            est_start = frac * total_dur
            est_end   = (frac + 1 / max(n_sent, 1)) * total_dur

            wx_conf = wx_conf_by_time(wx_segs, est_start, est_end)
            if wx_conf is None:
                wx_search = max(0, int(frac * len(wx_segs)) - 5)
                wx_s, wx_e = find_best_span_indices(anchor, wx_segs, wx_search)
                wx_conf = mean_conf_for_span(wx_segs, wx_s, wx_e)

            qw_search = max(0, int(frac * n_qw) - 5)
            qw_s, qw_e = find_best_span_indices(anchor, qw_segs, max(qw_cursor, qw_search))
            qw_conf = mean_conf_for_span(qw_segs, qw_s, qw_e)
            qw_cursor = qw_e or qw_cursor

            par_search = max(0, int(frac * n_par) - 5)
            par_s, par_e = find_best_span_indices(anchor, par_segs, max(par_cursor, par_search))
            par_conf = mean_conf_for_span(par_segs, par_s, par_e)
            par_cursor = par_e or par_cursor

            mean_conf = round((wx_conf + qw_conf + par_conf) / 3, 4)

            all_mean.append(mean_conf)
            all_wx.append(wx_conf)
            all_qw.append(qw_conf)
            all_par.append(par_conf)

            sentences.append({
                "sent_pos":          sent_pos,
                "confidence":        mean_conf,
                "whisperx_acoustic": round(wx_conf,  4),
                "qwen_acoustic":     round(qw_conf,  4),
                "parakeet_acoustic": round(par_conf, 4),
                "sentence":          anchor,
            })

        output_samples.append({"dataset_index": dataset_index, "sentences": sentences})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    corpus_wer = CORPUS_WER.get(dataset)

    _save(output_path, "acoustic_mean", dataset, corpus_wer, output_samples)

    for model, field in [("whisperx", "whisperx_acoustic"),
                          ("qwen",     "qwen_acoustic"),
                          ("parakeet", "parakeet_acoustic")]:
        variant_path    = os.path.join(OUTPUT_DIR, f"acoustic_{model}_{dataset}.json")
        variant_samples = [
            {"dataset_index": samp["dataset_index"],
             "sentences": [{**sent, "confidence": sent[field]} for sent in samp["sentences"]]}
            for samp in output_samples
        ]
        _save(variant_path, f"acoustic_{model}", dataset, corpus_wer, variant_samples)

    n = len(all_mean)
    if n:
        print(f"  Total sentences: {n}")
        print(f"  Mean acoustic (mean):     {sum(all_mean)/n:.3f}")
        print(f"  Mean acoustic (whisperx): {sum(all_wx)/n:.3f}")
        print(f"  Mean acoustic (qwen):     {sum(all_qw)/n:.3f}")
        print(f"  Mean acoustic (parakeet): {sum(all_par)/n:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=DATASETS + ["all"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        compute_dataset(dataset, rerun=args.rerun)


if __name__ == "__main__":
    main()