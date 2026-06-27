"""
src/selector.py

Shared infrastructure for all combination/selector scripts — dataset sizes,
canonical benchmark file paths, subset sampling, confidence flagging, and
Ollama selector calls. Centralises what was previously copy-pasted across
run_naive_combination_v3.py, run_context_selector.py, run_context_selector_v2.py,
run_context_selector_v1_confidence.py, run_context_selector_v2_confidence.py,
and run_naive_combination_confidence.py.

Usage:
    from src.selector import (
        get_subset_indices, load_qwen_samples, load_subset_by_sample_index,
        strip_lowconf_markers, DATASET_SIZES, CANONICAL_FILES, SUBSET_FILES,
        OLLAMA_MODELS, ollama_select,
    )
"""

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
    "english_dialects": 2543,
    "shetland":         100,
}

# ── Benchmark file paths ───────────────────────────────────────────────────────

# Full-dataset benchmark files (relative to project root)
CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "results/benchmarks/main/qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "results/benchmarks/main/qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "results/benchmarks/main/qwen_english_dialects_20260525_000627.json",
    ("qwen",     "shetland"):         "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",

    ("whisper",  "commonvoice"):      "results/benchmarks/main/whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "results/benchmarks/main/whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "results/benchmarks/main/whisper_english_dialects_20260525_110315.json",
    ("whisper",  "shetland"):         "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",

    ("parakeet", "commonvoice"):      "results/benchmarks/main/parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "results/benchmarks/main/parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "results/benchmarks/main/parakeet_english_dialects_20260524_234807.json",
    ("parakeet", "shetland"):         "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",

    ("wav2vec2", "commonvoice"):      "results/benchmarks/main/wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "results/benchmarks/main/wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "results/benchmarks/main/wav2vec2_english_dialects_20260526_073439.json",
    ("wav2vec2", "shetland"):         "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
}

# Per-word confidence subset files (150-sample subsets with segments)
SUBSETS_DIR = "results/benchmarks/subsets"

SUBSET_FILES = {
    ("whisper",  "commonvoice"):      "whisper_commonvoice_sub150.json",
    ("whisper",  "edacc"):            "whisper_edacc_sub150.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_sub150.json",
    ("whisper",  "shetland"):         "whisper_shetland_sub100.json",

    ("parakeet", "commonvoice"):      "parakeet_commonvoice_sub150.json",
    ("parakeet", "edacc"):            "parakeet_edacc_sub150.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_sub150.json",
    ("parakeet", "shetland"):         "parakeet_shetland_sub100.json",
}

# ── Ollama model names ─────────────────────────────────────────────────────────

OLLAMA_MODELS = {
    "qwen":    "qwen2.5:7b",
    "qwen14b": "qwen2.5:14b",
    "mistral": "mistral:7b",
    "gemma2":  "gemma2:9b",
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
    return load_samples(CANONICAL_FILES[("qwen", dataset)])


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