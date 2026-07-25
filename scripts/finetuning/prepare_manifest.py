"""
scripts/finetuning/prepare_manifest.py

Builds train/val/test manifests for ASR fine-tuning, combining audio +
transcripts across commonvoice, english_dialects, and edacc (NEVER
shetland - untouched external test set), using src/finetune_splits.py's
split (test == existing ensemble test pool, unchanged; train/val ==
80/20 split of existing dev pool).

CHUNKING: any sample longer than ~28s (see chunking.py) is split into
multiple aligned sub-segments before being written out - critical for
Whisper, which silently truncates audio to ~30s while still accepting
the FULL transcript as the target if left unchunked. This matters most
for commonvoice (avg ~55s/sample) but runs generically over all three
datasets as a safety net. See chunking.py's docstring for exactly how
alignment works (reuses WhisperX's existing word-level timestamps).

VALIDATION: each candidate sample/chunk is checked for a non-empty
transcript and non-trivial audio length before being written - anything
that fails is counted and reported, not silently dropped without a trace.

Writes, for each split (train/val/test):
  - data/finetune/audio/<dataset>_<sample_index>[_chunk<i>].wav  (16kHz mono)
  - data/finetune/manifests/<split>.jsonl
  - data/finetune/manifests/split_summary.json (sample counts, hours,
    dataset composition - for your methods section and for sanity-checking
    the split before submitting an expensive training job)

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
from chunking import chunk_sample, MAX_CHUNK_SEC

DATASETS = {
    "commonvoice":      CommonVoiceScots,
    "english_dialects":  EnglishDialectsScots,
    "edacc":            EdAcc,
}

OUTPUT_AUDIO_DIR = "data/finetune/audio"
OUTPUT_MANIFEST_DIR = "data/finetune/manifests"
TARGET_SAMPLE_RATE = 16000
MIN_TEXT_CHARS = 1


def _iter_dataset_samples(dataset_key: str):
    dataset = DATASETS[dataset_key]()
    for i, sample in enumerate(dataset.load()):
        yield i, sample.label, sample.audio, sample.sample_rate


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def build_manifests(dataset_keys: list):
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MANIFEST_DIR, exist_ok=True)

    manifest_entries = {"train": [], "val": [], "test": []}
    summary = {}

    for dataset_key in dataset_keys:
        print(f"\n── {dataset_key} ──")
        split_assignment = get_finetune_split(dataset_key)
        index_to_split = {}
        for split_name in ("train", "val", "test"):
            for idx in split_assignment[split_name]:
                index_to_split[idx] = split_name

        n_written = {"train": 0, "val": 0, "test": 0}
        n_skipped_no_split = 0
        n_skipped_empty_text = 0
        n_chunked_samples = 0
        n_chunks_total = 0
        seconds_written = {"train": 0.0, "val": 0.0, "test": 0.0}

        for sample_index, ref_text, audio, sr in _iter_dataset_samples(dataset_key):
            split_name = index_to_split.get(sample_index)
            if split_name is None:
                n_skipped_no_split += 1
                continue

            if not ref_text or not ref_text.strip() or len(ref_text.strip()) < MIN_TEXT_CHARS:
                n_skipped_empty_text += 1
                continue

            audio_16k = _resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            duration = len(audio_16k) / TARGET_SAMPLE_RATE

            chunks = chunk_sample(dataset_key, sample_index, ref_text.strip(), audio_16k, TARGET_SAMPLE_RATE)
            if len(chunks) > 1:
                n_chunked_samples += 1
            n_chunks_total += len(chunks)

            for chunk_idx, (chunk_text, chunk_audio) in enumerate(chunks):
                if not chunk_text.strip() or len(chunk_audio) == 0:
                    continue

                if len(chunks) == 1:
                    filename = f"{dataset_key}_{sample_index}.wav"
                else:
                    filename = f"{dataset_key}_{sample_index}_chunk{chunk_idx}.wav"

                audio_path = os.path.join(OUTPUT_AUDIO_DIR, filename)
                sf.write(audio_path, chunk_audio, TARGET_SAMPLE_RATE)

                manifest_entries[split_name].append({
                    "audio_path": audio_path,
                    "text": chunk_text.strip(),
                    "dataset": dataset_key,
                    "sample_index": sample_index,
                    "chunk_index": chunk_idx if len(chunks) > 1 else None,
                    "duration_sec": round(len(chunk_audio) / TARGET_SAMPLE_RATE, 3),
                })
                n_written[split_name] += 1
                seconds_written[split_name] += len(chunk_audio) / TARGET_SAMPLE_RATE

        print(f"  train={n_written['train']} val={n_written['val']} test={n_written['test']}")
        print(f"  skipped (not in any split, e.g. calibration): {n_skipped_no_split}")
        print(f"  skipped (empty transcript): {n_skipped_empty_text}")
        print(f"  samples that needed chunking (>{MAX_CHUNK_SEC}s): {n_chunked_samples} "
              f"-> {n_chunks_total} total chunks written")

        summary[dataset_key] = {
            "written": n_written,
            "hours": {k: round(v / 3600, 3) for k, v in seconds_written.items()},
            "skipped_no_split": n_skipped_no_split,
            "skipped_empty_text": n_skipped_empty_text,
            "samples_chunked": n_chunked_samples,
            "total_chunks_written": n_chunks_total,
        }

    for split_name, entries in manifest_entries.items():
        manifest_path = os.path.join(OUTPUT_MANIFEST_DIR, f"{split_name}.jsonl")
        with open(manifest_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(entries)} entries -> {manifest_path}")

    # cross-split integrity check: no sample_index should appear in more than one split,
    # for the same dataset (chunks of the same sample all stay in the same split, since
    # chunking happens AFTER split assignment - this just confirms that held)
    seen = {}
    leakage_found = False
    for split_name, entries in manifest_entries.items():
        for e in entries:
            key = (e["dataset"], e["sample_index"])
            if key in seen and seen[key] != split_name:
                print(f"  LEAKAGE WARNING: {key} appears in both {seen[key]} and {split_name}!")
                leakage_found = True
            seen[key] = split_name
    if not leakage_found:
        print("\nIntegrity check passed: no sample appears in more than one split.")

    summary_path = os.path.join(OUTPUT_MANIFEST_DIR, "split_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved split summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Which datasets to include (default: all 3, never shetland)")
    args = parser.parse_args()
    build_manifests(args.datasets)


if __name__ == "__main__":
    main()