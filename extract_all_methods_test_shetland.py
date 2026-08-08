"""
extract_all_methods_test_shetland.py

Pulls test-split + Shetland results for every pre-fine-tuning method,
in the requested format:

Method:
  Common Voice: severity, WER, N
  EdAcc: severity, WER, N
  English Dialects: severity, WER, N
  Shetland: severity, WER, N

Covers: Best individual model, ROVER, Pairwise-WER (MBR-style)
consensus, Candidate selection, Anchored fusion (V1), Unanchored
fusion. Missing files are reported explicitly rather than silently
skipped - some (e.g. Selection on Shetland) are known-and-expected
gaps by design; others may be genuine gaps worth checking.

Usage:
    python extract_all_methods_test_shetland.py
"""

import json
import os
import glob

from jiwer import wer as compute_wer
from src.text_normalise import normalise
from src.splits import get_indices_for_split

DATASETS = ["commonvoice", "edacc", "english_dialects"]
DATASET_DISPLAY = {"commonvoice": "Common Voice", "edacc": "EdAcc", "english_dialects": "English Dialects"}

BEST_MODEL_PER_DATASET = {
    "commonvoice": "parakeet",
    "edacc": "qwen",
    "english_dialects": "whisperx",
    "shetland": "qwen",
}


def show(dataset_label, sev, wer, n):
    if sev is None:
        print(f"  {dataset_label}: NOT FOUND")
    else:
        print(f"  {dataset_label}: severity={sev:.3f}, WER={wer*100:.2f}%, N={n}")


def from_file(path):
    if not path or not os.path.exists(path):
        return None, None, None
    data = json.load(open(path))
    return data.get("mean_severity"), data.get("corpus_wer"), data.get("num_samples")


def find_baseline_model_path(model, dataset):
    if dataset == "shetland":
        shetland_files = {
            "qwen":     "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
            "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
            "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
            "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
        }
        return shetland_files.get(model)
    matches = sorted(glob.glob(f"writeup_results/benchmarks/main/{model}_{dataset}_*.json"))
    matches = [m for m in matches if "sub150" not in m and "sub100" not in m
               and "whisper_ft_chunked" not in m]
    return matches[-1] if matches else None


def best_model_restricted(dataset, split):
    model = BEST_MODEL_PER_DATASET[dataset]
    path = find_baseline_model_path(model, dataset)
    if not path:
        return None, None, None
    data = json.load(open(path))
    samples = data.get("samples", [])
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i
    if split == "full":
        subset = samples
    else:
        idx_set = set(get_indices_for_split(dataset, split))
        subset = [s for s in samples if s.get("sample_index") in idx_set]
    refs, hyps, sevs = [], [], []
    for s in subset:
        if s.get("ref") and s.get("hyp"):
            refs.append(normalise(s["ref"]))
            hyps.append(normalise(s["hyp"]))
        if s.get("severity") is not None:
            sevs.append(s["severity"])
    wer_val = compute_wer(refs, hyps) if refs else None
    mean_sev = sum(sevs) / len(sevs) if sevs else None
    return mean_sev, wer_val, len(subset)


def run_method(name, path_fn, needs_restriction=False):
    print(f"\n{name}:")
    for d in DATASETS:
        if needs_restriction:
            sev, wer, n = path_fn(d, "test")
        else:
            sev, wer, n = from_file(path_fn(d, "test"))
        show(DATASET_DISPLAY[d], sev, wer, n)

    if needs_restriction:
        sev, wer, n = path_fn("shetland", "full")
    else:
        sev, wer, n = from_file(path_fn("shetland", "full"))
    show("Shetland", sev, wer, n)


# ── Best individual model ──
run_method("Best individual model", lambda d, s: best_model_restricted(d, s), needs_restriction=True)

# ── ROVER ──
run_method("ROVER", lambda d, s: (
    f"writeup_results/voting_calib_fixed/rover/rover_{d}_{s}.json" if s == "test"
    else "writeup_results/voting/rover/rover_shetland_full.json"
))

# ── Pairwise-WER (MBR-style) consensus ──
run_method("Pairwise-WER consensus (MBR-style)", lambda d, s: (
    f"writeup_results/ensembles_calib_fixed/mbr_consensus/mbr_{d}_{s}.json" if s == "test"
    else "writeup_results/ensembles/mbr_consensus/mbr_shetland_full.json"
))

# ── Candidate selection ──
run_method("Candidate selection", lambda d, s: (
    f"writeup_results/grid_calib_fixed/selection_naive/selection_naive_{d}_gemma4_{s}.json" if s == "test"
    else None  # confirmed: Selection was never run on Shetland
))

# ── Anchored fusion (V1) ──
run_method("Anchored fusion (V1)", lambda d, s: (
    f"writeup_results/grid_calib_fixed/anchored_correction_v1/anchored_correction_v1_{d}_gemma4_{s}.json" if s == "test"
    else "writeup_results/ensembles/context_v1/gemma4/context_shetland_gemma4_full.json"
))

# ── Unanchored fusion (baseline, pre-fine-tuning) ──
run_method("Unanchored fusion", lambda d, s: (
    f"writeup_results/grid_calib_fixed/unanchored_fusion_naive/unanchored_fusion_naive_{d}_gemma4_{s}.json" if s == "test"
    else "writeup_results/ensembles/naive/gemma4/naive_shetland_gemma4sel_full.json"
))