"""
scripts/finetuning/prepare_manifest.py

Builds train/val/test manifests for ASR fine-tuning, combining audio +
transcripts across commonvoice, english_dialects, and edacc (NEVER
shetland - untouched external test set), using src/finetune_splits.py's
70(56)/14/30-style split (test == existing ensemble test pool, unchanged;
train/val == 80/20 split of existing dev pool).

Writes, for each split (train/val/test):
  - data/finetune/audio/<dataset>_<sample_index>.wav   (16kHz mono wav)
  - data/finetune/manifests/<split>.jsonl               (one line per sample:
      {"audio_path": ..., "text": ..., "dataset": ..., "sample_index": ...})

Both the Whisper and wav2vec2 fine-tuning scripts consume these same
manifests, so data prep only needs to happen once regardless of which
model(s) you're training.

ASSUMPTIONS TO VERIFY AGAINST YOUR ACTUAL src/datasets.py:
  - dataset.load() yields objects with .label (str), .audio (np.ndarray),
    .sample_rate (int) - matching the pattern already used in
    src/benchmark.py's run_benchmark(). If your actual Dataset classes
    expose audio/label differently, adjust _iter_dataset_samples() below.
  - Audio arrays are assumed mono; if any dataset yields stereo, add a
    channel-averaging step before writing.

Usage:
    python scripts/finetuning/prepare_manifest.py
    python scripts/finetuning/prepare_manifest.py --datasets commonvoice edacc
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf

from src.datasets import EnglishDialectsScots, CommonVoiceScots, EdAcc
from src.finetune_splits import get_finetune_split

DATASETS = {
    "commonvoice":      CommonVoiceScots,
    "english_dialects":  EnglishDialectsScots,
    "edacc":            EdAcc,
}

OUTPUT_AUDIO_DIR = "data/finetune/audio"
OUTPUT_MANIFEST_DIR = "data/finetune/manifests"
TARGET_SAMPLE_RATE = 16000


def _iter_dataset_samples(dataset_key: str):
    """Yields (sample_index, ref_text, audio_array, sample_rate) for every
    sample in a dataset, in the same order/indexing run_benchmark.py uses
    (enumerate(dataset.load())), so sample_index lines up with what
    get_finetune_split() expects."""
    dataset = DATASETS[dataset_key]()
    for i, sample in enumerate(dataset.load()):
        yield i, sample.label, sample.audio, sample.sample_rate


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    # lazy import - only needed if resampling is actually required
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def build_manifests(dataset_keys: list):
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MANIFEST_DIR, exist_ok=True)

    manifest_entries = {"train": [], "val": [], "test": []}

    for dataset_key in dataset_keys:
        print(f"\n── {dataset_key} ──")
        split_assignment = get_finetune_split(dataset_key)
        # invert: sample_index -> split name, for O(1) lookup during iteration
        index_to_split = {}
        for split_name in ("train", "val", "test"):
            for idx in split_assignment[split_name]:
                index_to_split[idx] = split_name

        n_written = {"train": 0, "val": 0, "test": 0, "skipped": 0}

        for sample_index, ref_text, audio, sr in _iter_dataset_samples(dataset_key):
            split_name = index_to_split.get(sample_index)
            if split_name is None:
                # not in train/val/test - either excluded calibration ID,
                # or (shouldn't happen) an indexing mismatch worth investigating
                n_written["skipped"] += 1
                continue

            if not ref_text or not ref_text.strip():
                n_written["skipped"] += 1
                continue

            audio_16k = _resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            filename = f"{dataset_key}_{sample_index}.wav"
            audio_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
            sf.write(audio_path, audio_16k, TARGET_SAMPLE_RATE)

            manifest_entries[split_name].append({
                "audio_path": audio_path,
                "text": ref_text.strip(),
                "dataset": dataset_key,
                "sample_index": sample_index,
            })
            n_written[split_name] += 1

        print(f"  train={n_written['train']} val={n_written['val']} "
              f"test={n_written['test']} skipped={n_written['skipped']}")

    for split_name, entries in manifest_entries.items():
        manifest_path = os.path.join(OUTPUT_MANIFEST_DIR, f"{split_name}.jsonl")
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(entries)} entries -> {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Which datasets to include (default: all 3, never shetland)")
    args = parser.parse_args()
    build_manifests(args.datasets)


if __name__ == "__main__":
    main()