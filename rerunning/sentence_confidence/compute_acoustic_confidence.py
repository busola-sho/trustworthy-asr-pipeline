"""
rerunning/sentence_confidence/compute_acoustic_confidence.py

Method 3: Acoustic confidence.

UPDATED to work with naive (locked best technique) instead of the old
context_v2 + qwen2.5 pipeline - same changes as compute_crossmodel_agreement.py:
  - COMBO_FILES points at naive_probscore's output
  - Uses find_canonical_file() for current benchmark files, not the old
    static 150-sample subset paths
  - Extended from 3 models (whisperx/qwen/parakeet) to all 4
    (+ wav2vec2)
  - corpus_wer read from the combo file's own field, not a hardcoded dict

For each pipeline sentence:
  1. WhisperX: words in sentence time range (timestamp-based) -> mean confidence
  2. Qwen/Parakeet/wav2vec2: best matching span via constrained text search -> mean confidence
  3. mean_acoustic = mean across all 4 models

Also saves individual model variants:
  acoustic_whisperx_{dataset}.json / acoustic_qwen_{dataset}.json /
  acoustic_parakeet_{dataset}.json / acoustic_wav2vec2_{dataset}.json

Usage:
    python rerunning/sentence_confidence/compute_acoustic_confidence.py --dataset commonvoice
    python rerunning/sentence_confidence/compute_acoustic_confidence.py --dataset all
"""

import json
import os
import string
import difflib
import argparse
from typing import List, Dict

from src.selector import find_canonical_file, load_samples

OUTPUT_DIR = "writeup_results/sentence_confidence"

COMBO_FILES = {
    "commonvoice":      "writeup_results/ensembles/naive_probscore/naive_probscore_commonvoice_gemma4sel_dev.json",
    "edacc":            "writeup_results/ensembles/naive_probscore/naive_probscore_edacc_gemma4sel_dev.json",
    "english_dialects": "writeup_results/ensembles/naive_probscore/naive_probscore_english_dialects_gemma4sel_dev.json",
    "shetland":         "writeup_results/ensembles/naive_probscore/naive_probscore_shetland_gemma4sel_full.json",
}

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS   = ["commonvoice", "edacc", "english_dialects", "shetland"]


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


def load_model_samples(model: str, dataset: str) -> Dict[int, Dict]:
    try:
        path = find_canonical_file(model, dataset)
    except FileNotFoundError:
        print(f"  WARNING: no canonical file found for {model}/{dataset}")
        return {}
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def load_combo(dataset: str) -> tuple:
    path = COMBO_FILES.get(dataset)
    if not path or not os.path.exists(path):
        print(f"  ERROR: combo file not found: {path}")
        return [], None
    with open(path) as f:
        data = json.load(f)
    return data.get("samples", []), data.get("corpus_wer")


def _save(path, method, dataset, corpus_wer, samples):
    with open(path, "w") as f:
        json.dump({"method": method, "dataset": dataset,
                   "corpus_wer": corpus_wer, "samples": samples},
                  f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def compute_dataset(dataset, rerun=False):
    print(f"\n── Acoustic confidence: {dataset} ──")

    output_path = os.path.join(OUTPUT_DIR, f"acoustic_confidence_{dataset}.json")
    if os.path.exists(output_path) and not rerun:
        print(f"  Already exists. Use --rerun to recompute.")
        return

    combo_samples, corpus_wer = load_combo(dataset)
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
        sent_confs    = s.get("sentence_confidences", [])
        if dataset_index is None or not sent_confs:
            continue

        model_segs = {m: segs_by_model[m].get(dataset_index, {}).get("segments", []) for m in ASR_MODELS}
        n_segs     = {m: len(model_segs[m]) for m in ASR_MODELS}
        n_sent     = len(sent_confs)

        sentences = []
        cursors   = {m: 0 for m in ASR_MODELS}
        total_dur = max((model_segs["whisperx"][-1].get("end") or 0) if model_segs["whisperx"] else 0, 0.1)

        for sent_pos, sc in enumerate(sent_confs):
            anchor = sc.get("sentence", "").strip()
            if not anchor:
                continue

            frac      = sent_pos / max(n_sent, 1)
            est_start = frac * total_dur
            est_end   = (frac + 1 / max(n_sent, 1)) * total_dur

            model_confs = {}

            wx_conf = wx_conf_by_time(model_segs["whisperx"], est_start, est_end)
            if wx_conf is None:
                wx_search = max(0, int(frac * n_segs["whisperx"]) - 5)
                wx_s, wx_e = find_best_span_indices(anchor, model_segs["whisperx"], wx_search)
                wx_conf = mean_conf_for_span(model_segs["whisperx"], wx_s, wx_e)
            model_confs["whisperx"] = wx_conf

            for m in ["qwen", "parakeet", "wav2vec2"]:
                search = max(0, int(frac * n_segs[m]) - 5)
                m_s, m_e = find_best_span_indices(anchor, model_segs[m], max(cursors[m], search))
                model_confs[m] = mean_conf_for_span(model_segs[m], m_s, m_e)
                cursors[m] = m_e or cursors[m]

            mean_conf = round(sum(model_confs.values()) / len(model_confs), 4)
            all_mean.append(mean_conf)
            for m in ASR_MODELS:
                all_by_model[m].append(model_confs[m])

            sentences.append({
                "sent_pos":   sent_pos,
                "confidence": mean_conf,
                **{f"{m}_acoustic": round(model_confs[m], 4) for m in ASR_MODELS},
                "sentence":   anchor,
            })

        output_samples.append({"dataset_index": dataset_index, "sentences": sentences})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save(output_path, "acoustic_mean", dataset, corpus_wer, output_samples)

    for model in ASR_MODELS:
        field = f"{model}_acoustic"
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
        print(f"  Mean acoustic (mean): {sum(all_mean)/n:.3f}")
        for m in ASR_MODELS:
            print(f"  Mean acoustic ({m}): {sum(all_by_model[m])/n:.3f}")


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