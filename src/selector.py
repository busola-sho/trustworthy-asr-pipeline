"""
src/selector.py

Shared infrastructure for all combination/selector scripts — dataset sizes,
canonical benchmark file paths, subset sampling, confidence flagging, and
Ollama selector calls. Centralises what was previously copy-pasted across
run_naive_combination_v3.py, run_context_selector.py, run_context_selector_v2.py,
run_context_selector_v1_confidence.py, run_context_selector_v2_confidence.py,
and run_naive_combination_confidence.py.

CHANGES:
  - "whisper" replaced with "whisperx" everywhere (standing decision:
    WhisperX replaces plain Whisper across all ensemble techniques)
  - CANONICAL_FILES (static, hardcoded timestamps) replaced with
    find_canonical_file() - a dynamic lookup that finds the most recent
    FULL-dataset benchmark file for a given model/dataset, searching
    writeup_results/benchmarks/main/ first, falling back to
    results/benchmarks/main/. This avoids hardcoding filenames that go
    stale every time a benchmark is rerun, and fixes the old dict pointing
    at 150-sample subset files rather than full-dataset runs.
  - OLLAMA_MODELS extended with the locked judge-shortlist models
    (phi4, ministral3, gemma4, qwen3.5), for use in the selector ablation.

Usage:
    from src.selector import (
        get_subset_indices, load_qwen_samples, load_subset_by_sample_index,
        strip_lowconf_markers, DATASET_SIZES, find_canonical_file, SUBSET_FILES,
        OLLAMA_MODELS, ollama_select,
    )
"""

import glob
import json
import os
import random
import re
from ollama import Client

# ── Dataset configuration ──────────────────────────────────────────────────────

SEED     = 42
N_SUBSET = 150

DATASET_SIZES = {
    "commonvoice":      680,
    "edacc":            198,
    "shetland":         100,
    "english_dialects": 2543,
}

# ── Benchmark file paths ───────────────────────────────────────────────────────

# Full-dataset benchmark files are timestamped and rerun periodically, so we
# look them up dynamically rather than hardcoding filenames that go stale.
# Searches writeup_results/ (current) first, then results/ (legacy) as fallback.
BENCHMARK_SEARCH_DIRS = [
    "writeup_results/benchmarks/main",
    "results/benchmarks/main",
]

# Shetland is a small, fixed, fully-used held-out set (100 samples, never
# re-subsetted) - these paths are stable and kept as a static lookup rather
# than searched for, since there's only ever one canonical file per model.
SHETLAND_FILES = {
    "qwen":     "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
    "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
    "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
    "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
}


def find_canonical_file(model: str, dataset: str) -> str:
    """
    Find the most recent FULL-dataset benchmark file for a given model and
    dataset. Excludes 150/100-sample subset files (only wants full runs).
    Searches writeup_results/benchmarks/main/ first, falls back to
    results/benchmarks/main/. Shetland uses its own static lookup instead,
    since it's a small fixed set with no full/subset distinction.

    Raises FileNotFoundError with a clear message if nothing is found -
    this is deliberate: silently falling back to a stale or wrong file
    would be worse than failing loudly here.
    """
    if dataset == "shetland":
        if model not in SHETLAND_FILES:
            raise FileNotFoundError(f"No Shetland file registered for model={model}")
        return SHETLAND_FILES[model]

    for base_dir in BENCHMARK_SEARCH_DIRS:
        pattern = os.path.join(base_dir, f"{model}_{dataset}_*.json")
        matches = sorted(glob.glob(pattern))
        matches = [m for m in matches if "sub150" not in m and "sub100" not in m]
        if matches:
            return matches[-1]   # filenames are timestamp-sorted, so latest sorts last

    raise FileNotFoundError(
        f"No full-dataset benchmark file found for model={model} dataset={dataset} "
        f"in {BENCHMARK_SEARCH_DIRS}. Run the benchmark/transcription script for "
        f"this model/dataset first, or check the filename pattern matches "
        f"'{model}_{dataset}_<timestamp>.json'."
    )


# Per-word confidence subset files (150-sample subsets with segments) - unchanged,
# these are only used by the sentence-confidence pipeline, not the ensemble scripts.
SUBSETS_DIR = "results/benchmarks/subsets"

SUBSET_FILES = {
    ("whisperx", "commonvoice"):      "whisper_commonvoice_sub150.json",
    ("whisperx", "edacc"):            "whisper_edacc_sub150.json",
    ("whisperx", "english_dialects"): "whisper_english_dialects_sub150.json",
    ("whisperx", "shetland"):         "shetland/whisper_shetland_sub100.json",

    ("parakeet", "commonvoice"):      "parakeet_commonvoice_sub150.json",
    ("parakeet", "edacc"):            "parakeet_edacc_sub150.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_sub150.json",
    ("parakeet", "shetland"):         "shetland/parakeet_shetland_sub100.json",

    ("qwen",     "commonvoice"):      "qwen3asr_commonvoice_sub150.json",
    ("qwen",     "edacc"):            "qwen3asr_edacc_sub150.json",
    ("qwen",     "english_dialects"): "qwen3asr_english_dialects_sub150.json",
    ("qwen",     "shetland"):         "shetland/qwen3asr_shetland_sub100.json",
}

# ── Ollama model names ─────────────────────────────────────────────────────────

OLLAMA_MODELS = {
    # legacy / original selector candidates
    "qwen":      "qwen2.5:7b",
    "qwen14b":   "qwen2.5:14b",
    "mistral":   "mistral:7b",
    "gemma2":    "gemma2:9b",
    # locked judge-shortlist models, added for the selector ablation -
    # confirmed working Ollama tags from the judge calibration work
    "phi4":       "phi4:14b",
    "ministral3": "ministral-3:14b",
    "gemma4":     "gemma4:12b",
    "qwen3.5":    "qwen3.5:9b",
}

# ── Subset sampling ────────────────────────────────────────────────────────────

def get_subset_indices(dataset: str) -> list:
    """
    Return the 150-sample subset indices for a dataset (seed=42).
    For datasets with <= 150 samples (e.g. Shetland at 100), returns
    all indices without sampling.
    """
    n_total = DATASET_SIZES[dataset]
    if n_total <= N_SUBSET:
        return list(range(n_total))
    random.seed(SEED)
    return sorted(random.sample(range(n_total), N_SUBSET))

# ── File loading ───────────────────────────────────────────────────────────────

def load_samples(path: str) -> list:
    """Load samples list from a benchmark JSON file."""
    with open(path) as f:
        return json.load(f)["samples"]


def load_qwen_samples(dataset: str) -> list:
    """Load Qwen3-ASR samples for a given dataset."""
    return load_samples(find_canonical_file("qwen", dataset))


def load_subset_by_sample_index(model: str, dataset: str) -> dict:
    """
    Load a confidence subset file and return a dict mapping
    sample_index -> sample dict for O(1) lookup by original dataset index.
    """
    path = os.path.join(SUBSETS_DIR, SUBSET_FILES[(model, dataset)])
    with open(path) as f:
        data = json.load(f)
    return {
        s["sample_index"]: s
        for s in data.get("samples", [])
        if s.get("sample_index") is not None
    }

# ── Confidence flagging ────────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD = 0.7


def build_flagged_transcript(segments: list, threshold: float = DEFAULT_CONF_THRESHOLD) -> str:
    """
    Wrap words below confidence threshold with [LOW-CONF: word] markers.
    Returns empty string if segments is None or empty.
    """
    if not segments:
        return ""
    parts = []
    for seg in segments:
        word = seg.get("word", "")
        conf = seg.get("confidence")
        if not word:
            continue
        if conf is not None and conf < threshold:
            parts.append(f"[LOW-CONF: {word}]")
        else:
            parts.append(word)
    return " ".join(parts)


def get_transcript_for_selector(
    hyp: str,
    segments: list,
    threshold: float = DEFAULT_CONF_THRESHOLD,
) -> str:
    """
    Return flagged transcript if segments are available and have confidence
    scores, otherwise return the plain hyp unchanged (e.g. for Qwen).
    """
    if not segments:
        return hyp
    if not any(seg.get("confidence") is not None for seg in segments):
        return hyp
    return build_flagged_transcript(segments, threshold=threshold)


def count_flagged_words(segments: list, threshold: float = DEFAULT_CONF_THRESHOLD) -> int:
    """Count words below the confidence threshold."""
    if not segments:
        return 0
    return sum(
        1 for seg in segments
        if seg.get("confidence") is not None and seg["confidence"] < threshold
    )


def strip_lowconf_markers(text: str) -> str:
    """
    Safety net: strip any [LOW-CONF: word] markers from selector output,
    keeping just the word. Prevents markers from corrupting WER computation.
    """
    return re.sub(r'\[LOW-CONF:\s*([^\]]+)\]', r'\1', text)


# ── Ollama selector call ───────────────────────────────────────────────────────

def ollama_select(
    client: Client,
    model_name: str,
    prompt: str,
    sleep: float = 0.05,
):
    """
    Send a selector prompt to Ollama and return the response text.
    Returns None on failure.
    """
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


def check_selector_available(client: Client, selector_key: str) -> bool:
    """Check whether the requested selector model is pulled and available."""
    model_name = OLLAMA_MODELS.get(selector_key)
    if not model_name:
        return False
    try:
        available = [m.model for m in client.list().models]
        return any(model_name in m for m in available)
    except Exception:
        return False