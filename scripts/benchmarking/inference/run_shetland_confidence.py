"""
run_shetland_confidence.py

Extracts per-word confidence segments for Whisper and Parakeet on the
Shetland dataset, using the same src.models classes (with logprob extraction)
as the main run_benchmark.py. Saves output in the same format as
results/benchmarks/subsets/whisper_commonvoice_sub150.json etc., so the
existing confidence-flagging selector scripts can use them directly.

Output:
    results/benchmarks/subsets/whisper_shetland_sub100.json
    results/benchmarks/subsets/parakeet_shetland_sub100.json

Usage:
    python scripts/benchmarking/inference/run_shetland_confidence.py --model whisper
    python scripts/benchmarking/inference/run_shetland_confidence.py --model parakeet
    python scripts/benchmarking/inference/run_shetland_confidence.py --model all
"""

import argparse
import json
import os
import librosa
import numpy as np
import pandas as pd
from jiwer import wer
import re

from src.models import Whisper, Parakeet

DATA_DIR   = "data/shetland"
AUDIO_DIR  = os.path.join(DATA_DIR, "audios")
EXCEL_PATH = os.path.join(DATA_DIR, "shetland.xlsx")
OUTPUT_DIR = "results/benchmarks/subsets"

MODELS = {
    "whisper":  Whisper,
    "parakeet": Parakeet,
}

MODEL_NAMES = {
    "whisper":  "openai/whisper-large-v3",
    "parakeet": "nvidia/parakeet-ctc-1.1b",
}


def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()


def segment_to_dict(seg) -> dict:
    return {
        "word":       seg.word,
        "confidence": seg.confidence,
        "start":      seg.start,
        "end":        seg.end,
    }


def run_model(model_key: str, rerun: bool = False):
    output_path = os.path.join(OUTPUT_DIR, f"{model_key}_shetland_sub100.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"Resuming {model_key}/shetland from sample {start_from}/100")
    else:
        results    = []
        start_from = 0

    print(f"\nLoading {model_key}...")
    model_cls = MODELS[model_key]
    model     = model_cls(MODEL_NAMES[model_key])
    model.load()
    print(f"  {model_key} loaded on {model.device}")

    df = pd.read_excel(EXCEL_PATH, dtype={"clip_id": str, "audio_file": str})
    print(f"  Loaded {len(df)} samples from {EXCEL_PATH}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pos, (_, row) in enumerate(df.iterrows()):
        if pos < start_from:
            continue

        clip_id    = row["clip_id"]
        ref        = str(row["transcript"]).strip()
        audio_file = str(row["audio_file"]).strip()
        audio_path = os.path.join(AUDIO_DIR, audio_file)

        if not os.path.exists(audio_path):
            print(f"  WARNING: missing audio {audio_path} — skipping")
            results.append({
                "sample_index": pos, "clip_id": clip_id,
                "ref": ref, "hyp": None,
                "sample_WER": None, "segments": [],
                "error": "missing_audio",
            })
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"  ERROR loading {audio_path}: {e}")
            results.append({
                "sample_index": pos, "clip_id": clip_id,
                "ref": ref, "hyp": None,
                "sample_WER": None, "segments": [],
                "error": str(e),
            })
            continue

        try:
            transcription = model.transcribe(audio, sample_rate=16000)
            hyp      = transcription.text
            segments = [segment_to_dict(s) for s in transcription.segments]
        except Exception as e:
            print(f"  ERROR transcribing {clip_id}: {e}")
            results.append({
                "sample_index": pos, "clip_id": clip_id,
                "ref": ref, "hyp": None,
                "sample_WER": None, "segments": [],
                "error": str(e),
            })
            continue

        sample_wer_val = wer(normalise(ref), normalise(hyp)) if ref else 0.0

        results.append({
            "sample_index": pos,
            "clip_id":      clip_id,
            "ref":          ref,
            "hyp":          hyp,
            "sample_WER":   sample_wer_val,
            "segments":     segments,
        })

        if (pos + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump({"progress": len(results), "samples": results}, f,
                          indent=2, ensure_ascii=False)
            print(f"  {pos+1}/100 done")

    valid = [r for r in results if r.get("sample_WER") is not None and not r.get("error")]
    corpus_wer_val = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "model":       model_key,
        "dataset":     "shetland",
        "num_samples": len(valid),
        "corpus_wer":  corpus_wer_val,
        "samples":     results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,
                        choices=["whisper", "parakeet", "all"])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    models = ["whisper", "parakeet"] if args.model == "all" else [args.model]
    for m in models:
        run_model(m, rerun=args.rerun)


if __name__ == "__main__":
    main()