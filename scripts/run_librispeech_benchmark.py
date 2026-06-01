"""
run_librispeech_benchmark.py

Runs all 4 ASR models on LibriSpeech test-clean to verify WER against
published benchmarks. No MAR evaluation — WER only.

Usage:
    python scripts/run_librispeech_benchmark.py --model whisper
    python scripts/run_librispeech_benchmark.py --model wav2vec2
    python scripts/run_librispeech_benchmark.py --model parakeet
    python scripts/run_librispeech_benchmark.py --model qwen3asr
    python scripts/run_librispeech_benchmark.py --model all
"""
# # Patch CVE-2025-32434 check before any imports
# import transformers.utils.import_utils as _iu
# _iu.check_torch_load_is_safe = lambda: None

# # Also patch modeling_utils directly
# import transformers.modeling_utils as _mu
# _mu.check_torch_load_is_safe = lambda: None

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
from datasets import load_dataset
from jiwer import wer


# ── Add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Whisper, Wav2Vec2, Parakeet, Qwen3ASR

# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = "benchmarks"

MODELS = {
    "whisper":   Whisper,
    "wav2vec2":  Wav2Vec2,
    "parakeet":  Parakeet,
    "qwen3asr":  Qwen3ASR,
}

# Published LibriSpeech test-clean WERs for reference
PUBLISHED_WER = {
    "whisper":  2.7,
    "wav2vec2": 1.8,
    "parakeet": 1.83,
    "qwen3asr": 1.63,
}

# ── Text normalisation (Whisper-style, as used by Open ASR Leaderboard) ────────

def normalise(text: str) -> str:
    """Basic normalisation matching Open ASR Leaderboard evaluation."""
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_model(model_key: str, max_samples: int = None):
    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"Dataset: LibriSpeech test-clean")
    print(f"{'='*60}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"librispeech_{model_key}_{timestamp}.json")

    # load model
    print(f"Loading {model_key}...")
    model = MODELS[model_key]()
    model.load()
    print("Model loaded.")

    # load dataset
    print("Loading LibriSpeech test-clean...")
    dataset = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    print("Dataset loaded.")

    results    = []
    all_refs   = []
    all_hyps   = []

    for i, row in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        audio_array = np.array(row["audio"]["array"], dtype=np.float32)
        sample_rate = row["audio"]["sampling_rate"]
        label       = row["text"]

        try:
            transcription = model.transcribe(audio_array, sample_rate)
            hyp = transcription.text
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            hyp = ""

        ref_norm = normalise(label)
        hyp_norm = normalise(hyp)
        sample_wer = wer(ref_norm, hyp_norm) if ref_norm else 0.0

        results.append({
            "ref":        label,
            "hyp":        hyp,
            "sample_WER": sample_wer,
        })
        all_refs.append(ref_norm)
        all_hyps.append(hyp_norm)

        # save after every sample for resume safety
        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f, indent=2)

        if (i + 1) % 50 == 0:
            running_wer = wer(all_refs, all_hyps)
            print(f"  {i+1} samples | running WER: {running_wer*100:.2f}%")

    # final metrics
    corpus_wer = wer(all_refs, all_hyps)
    published  = PUBLISHED_WER.get(model_key, "N/A")

    output = {
        "model":         model.model_name,
        "dataset":       "librispeech_test_clean",
        "corpus_wer":    corpus_wer,
        "num_samples":   len(results),
        "published_wer": published,
        "gap":           round(corpus_wer * 100 - published, 2) if isinstance(published, float) else "N/A",
        "samples":       results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults:")
    print(f"  Corpus WER:    {corpus_wer*100:.2f}%")
    print(f"  Published WER: {published}%")
    print(f"  Gap:           {output['gap']}pp")
    print(f"  Saved to:      {output_path}")

    return {
        "model":         model_key,
        "corpus_wer":    corpus_wer * 100,
        "published_wer": published,
        "gap":           output["gap"],
        "n":             len(results),
    }

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples (for testing)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    summary = []
    for model_key in models_to_run:
        result = run_model(model_key, args.max_samples)
        summary.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Our WER':>10} {'Published':>10} {'Gap':>8} {'N':>6}")
    for r in summary:
        print(f"{r['model']:<12} {r['corpus_wer']:>9.2f}% "
              f"{str(r['published_wer']):>9}% "
              f"{str(r['gap']):>7}pp "
              f"{r['n']:>6}")

if __name__ == "__main__":
    main()