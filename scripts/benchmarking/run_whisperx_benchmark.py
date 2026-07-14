"""
scripts/benchmarking/run_whisperx_benchmark.py

Run WhisperX on benchmark datasets and save results in the same format
as existing subset files, so they can be used in existing selector/ROVER
experiments as a drop-in replacement for standard Whisper.

Reuses src/datasets.py for audio loading — no audio paths needed.

Usage:
    python scripts/benchmarking/run_whisperx_benchmark.py --dataset commonvoice
    python scripts/benchmarking/run_whisperx_benchmark.py --dataset edacc
    python scripts/benchmarking/run_whisperx_benchmark.py --dataset english_dialects
"""

import json
import os
import argparse
import time
import numpy as np
import torch
import whisperx
from jiwer import wer

from src.judge import normalise
from src.selector import get_subset_indices, SUBSETS_DIR
from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc, Shetland
from src.judge import normalise, ollama_mar
from ollama import Client

OLLAMA_HOST = "http://localhost:11434"

DATASETS = {
    "commonvoice":      CommonVoiceScots,
    "english_dialects": EnglishDialectsScots,
    "edacc":            EdAcc,
    "shetland":         Shetland,
}

WHISPERX_MODEL_SIZE = "large-v3"


def get_device():
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def run_dataset(dataset_name: str, max_samples: int = None, rerun: bool = False):
    print(f"\n── WhisperX benchmark: {dataset_name} ──")

    output_path = os.path.join(SUBSETS_DIR, f"whisperx_{dataset_name}_sub150.json")

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        n_done = len(existing.get("samples", []))
        print(f"  Already exists with {n_done} samples. Use --rerun to redo.")
        return

    indices = get_subset_indices(dataset_name)
    if max_samples:
        indices = indices[:max_samples]
    indices_set = set(indices)

    print(f"  {len(indices)} samples to transcribe")

    # load WhisperX models once
    client = Client(host=OLLAMA_HOST)
    device, compute_type = get_device()
    print(f"  Device: {device} ({compute_type})")
    print(f"  Loading WhisperX {WHISPERX_MODEL_SIZE}...")

    model = whisperx.load_model(
        WHISPERX_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
        language="en",
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en", device=device
    )
    print("  Models ready.")

    # load dataset and iterate
    dataset_cls = DATASETS[dataset_name]
    dataset     = dataset_cls()

    samples    = []
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}
    results    = {}  # sample_index -> result
    start_time = time.time()

    print("  Iterating dataset...")
    for global_idx, sample in enumerate(dataset.load()):
        if global_idx not in indices_set:
            continue

        ref = sample.label
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results[global_idx] = {
                "sample_index": global_idx,
                "ref":          ref,
                "hyp":          None,
                "sample_WER":   None,
                "segments":     [],
                "skipped":      True,
            }
            continue

        # resample to 16kHz if needed
        audio = sample.audio
        sr    = sample.sample_rate
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = audio.astype(np.float32)

        try:
            # transcribe
            wx_result = model.transcribe(audio, batch_size=4, language="en")

            # word-level alignment
            wx_result = whisperx.align(
                wx_result["segments"], align_model, align_metadata,
                audio, device, return_char_alignments=False,
            )

            # extract words + confidence scores
            word_segments = []
            full_words    = []
            for seg in wx_result.get("segments", []):
                for word_info in seg.get("words", []):
                    word = word_info.get("word", "").strip()
                    if word:
                        full_words.append(word)
                        word_segments.append({
                            "word":       word,
                            "confidence": word_info.get("score", 0.8),
                            "start":      word_info.get("start"),
                            "end":        word_info.get("end"),
                        })

            hyp        = " ".join(full_words)
            sample_wer = wer(normalise(ref), normalise(hyp))
            verdict    = ollama_mar(client, ref, hyp, sample_wer)

            results[global_idx] = {
                "sample_index":   global_idx,
                "ref":            ref,
                "hyp":            hyp,
                "sample_WER":     sample_wer,
                "qwen_verdict_p2":verdict,
                "segments":       word_segments,
            }

            pos = idx_to_pos.get(global_idx, -1)
            if (len(results)) % 10 == 0:
                print(f"  {len(results)}/{len(indices)} done "
                      f"({time.time()-start_time:.0f}s elapsed)")
                # save progress
                ordered = [results[i] for i in indices if i in results]
                with open(output_path, "w") as f:
                    json.dump({
                        "model": "whisperx", "dataset": dataset_name,
                        "samples": ordered
                    }, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ERROR sample {global_idx}: {e}")
            results[global_idx] = {
                "sample_index": global_idx,
                "ref":          ref,
                "hyp":          "",
                "sample_WER":   None,
                "segments":     [],
                "error":        str(e),
            }

        # stop early if we have all needed indices
        if len(results) >= len(indices):
            break

    # build ordered output
    ordered = [results[i] for i in indices if i in results]
    valid   = [s for s in ordered if s.get("sample_WER") is not None]

    corpus_wer_val = wer(
        [normalise(s["ref"]) for s in valid],
        [normalise(s["hyp"]) for s in valid],
    ) if valid else None

    mar = (
        sum(1 for s in valid if s.get("qwen_verdict_p2")) / len(valid)
        if valid else None
    )

    output = {
        "model":                   "whisperx",
        "dataset":                 dataset_name,
        "model_size":              WHISPERX_MODEL_SIZE,
        "corpus_wer":              corpus_wer_val,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "subset_indices":          indices,
        "samples":                 ordered,
    }

    os.makedirs(SUBSETS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="commonvoice",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    run_dataset(args.dataset,
                max_samples=args.max_samples,
                rerun=args.rerun)


if __name__ == "__main__":
    main()