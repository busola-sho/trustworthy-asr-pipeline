"""
test_context_injection.py

Tests whether passing a Scottish dialect glossary as context to Qwen3-ASR
improves transcription on hard dialect samples.

Runs the same audio twice — once without context, once with — and compares
WER and the specific dialect words recovered.

Usage:
    python scripts/benchmarking/inference/test_context_injection.py --audio data/shetland/audios/sland_s1_001.m4a --ref "reference transcript here"
    python scripts/benchmarking/inference/test_context_injection.py --n 10  # run on first 10 error samples
"""

import argparse
import json
import librosa
import numpy as np
import pandas as pd
from jiwer import wer
from src.judge import normalise

SCOTTISH_CONTEXT = """You are transcribing Scottish English speech. The speaker may use Scottish dialect words and expressions. Common examples:
- wee (small), braw (good/fine), dreich (dull/miserable weather)
- doesnae (doesn't), wisnae (wasn't), dinnae (didn't), cannae (can't), hasnae (hasn't)
- aye (yes), naw/nae (no), och (oh), eh (um/yes)
- ken (know), tae (to), wae/wi (with), fae/frae (from)
- fa/fa' (who/fall), farr/whaur (where), abody/aabody (everybody)
- a'/aa (all), noo (now), aboot (about), oot (out), doon (down), awa (away)
- bairn (child), loch (lake), burn (stream), glen (valley)
- wane/waning (declining), syne (since/ago), aye (always/yes)
Preserve dialect words as spoken rather than normalising to standard English."""


def load_model():
    from src.models import Qwen3ASR
    model = Qwen3ASR()
    model.load()
    return model


def transcribe_both(model, audio, sample_rate, ref):
    result_plain = model.transcribe(audio, sample_rate, context="")
    result_ctx   = model.transcribe(audio, sample_rate, context=SCOTTISH_CONTEXT)

    wer_plain = wer(normalise(ref), normalise(result_plain.text))
    wer_ctx   = wer(normalise(ref), normalise(result_ctx.text))

    return {
        "ref":            ref,
        "plain":          result_plain.text,
        "context":        result_ctx.text,
        "segments_plain": result_plain.segments,
        "segments_ctx":   result_ctx.segments,
        "wer_plain":      wer_plain,
        "wer_ctx":        wer_ctx,
        "delta":          wer_ctx - wer_plain,
    }


def run_on_audio(model, audio_path, ref):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    result = transcribe_both(model, audio, ref)

    print(f"\n{'='*70}")
    print(f"REF:     {result['ref'][:200]}")
    print(f"PLAIN:   {result['plain'][:200]}  (WER={result['wer_plain']*100:.1f}%)")
    print(f"CONTEXT: {result['context'][:200]}  (WER={result['wer_ctx']*100:.1f}%)")
    delta = result['delta'] * 100
    direction = "IMPROVED" if delta < 0 else "WORSE" if delta > 0 else "SAME"
    print(f"DELTA: {delta:+.1f}pp  [{direction}]")
    return result


def run_on_error_samples(model, dataset, n):
    """Run on the hardest remaining errors from CommonVoice."""
    error_path = f"results/combinations_v2judge/context_v2_confidence/context_v2conf_{dataset}_qwen_t080_sub150.json"
    main_path  = f"results/benchmarks/main/qwen_{dataset}_20260524_153426.json" if dataset == "commonvoice" else None

    with open(error_path) as f:
        data = json.load(f)

    errors = [
        s for s in data["samples"]
        if s.get("qwen_verdict_p2") is True
        and not s.get("skipped") and not s.get("error")
    ]
    errors.sort(key=lambda s: s.get("sample_WER", 0), reverse=True)
    errors = errors[:n]

    if dataset == "commonvoice":
        audio_dir = "data/common-voice-scots/audios"
        # load reference from main benchmark file to get audio filename
        # CommonVoice stores audio filename in dataset — need to load dataset
        from src.datasets import CommonVoiceScots
        ds = CommonVoiceScots()
        samples = list(ds.load())

    results = []
    for e in errors:
        idx = e.get("dataset_index")
        if idx is None or idx >= len(samples):
            continue
        sample = samples[idx]
        ref    = e["ref"]

        print(f"\nSample {idx}...")
        result = transcribe_both(model, sample.audio, sample.sample_rate, ref)
        result["dataset_index"] = idx
        results.append(result)

        print(f"REF:     {ref}")
        print(f"PLAIN:   {result['plain']}  WER={result['wer_plain']*100:.1f}%")
        print(f"CONTEXT: {result['context']}  WER={result['wer_ctx']*100:.1f}%")
        delta = result['delta'] * 100
        print(f"DELTA: {delta:+.1f}pp  [{'IMPROVED' if delta < 0 else 'WORSE' if delta > 0 else 'SAME'}]")

    # summary
    improved = sum(1 for r in results if r["delta"] < 0)
    worse    = sum(1 for r in results if r["delta"] > 0)
    same     = sum(1 for r in results if r["delta"] == 0)
    avg_delta = sum(r["delta"] for r in results) / len(results) * 100 if results else 0

    print(f"\n{'='*50}")
    print(f"SUMMARY over {len(results)} hard samples:")
    print(f"  Improved: {improved}  Worse: {worse}  Same: {same}")
    print(f"  Average WER delta: {avg_delta:+.2f}pp")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",   default=None, help="Path to a single audio file")
    parser.add_argument("--ref",     default=None, help="Reference transcript for single audio test")
    parser.add_argument("--n",       type=int, default=10, help="Number of error samples to test")
    parser.add_argument("--dataset", default="commonvoice",
                        choices=["commonvoice", "edacc", "english_dialects"])
    args = parser.parse_args()

    print("Loading Qwen3-ASR...")
    model = load_model()

    if args.audio and args.ref:
        run_on_audio(model, args.audio, args.ref)
    else:
        run_on_error_samples(model, args.dataset, args.n)


if __name__ == "__main__":
    main()