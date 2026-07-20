"""
rerunning/run_whisperx_transcribe.py

PHASE 1 of 2: Run WhisperX transcription ONLY - no severity judging here.
This avoids running WhisperX and Phi-4 simultaneously, which was causing
memory contention/reload-thrashing on a 16GB machine (one 269s spike out
of otherwise 30-65s samples, traced to Phi-4 idling out and cold-reloading
mid-run).

Once this finishes, run PHASE 2 separately:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

Writes to BOTH locations for full tracking:
  - writeup_results/benchmarks/main/whisperx_{dataset}_{timestamp}.json  (new)
  - results/benchmarks/main/whisperx_{dataset}_{timestamp}.json          (old, kept for continuity)

Usage:
    python rerunning/run_whisperx_transcribe.py --dataset commonvoice --full
    python rerunning/run_whisperx_transcribe.py --dataset edacc --full
    python rerunning/run_whisperx_transcribe.py --dataset english_dialects --full
"""

import json
import os
import argparse
import time
import numpy as np
import torch
import whisperx
from jiwer import wer

from src.judge import normalise, is_tag_only
from src.selector import get_subset_indices, SUBSETS_DIR
from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc, Shetland
from datetime import datetime

DATASETS = {
    "commonvoice":      CommonVoiceScots,
    "english_dialects": EnglishDialectsScots,
    "edacc":            EdAcc,
    "shetland":         Shetland,
}

# Full dataset sizes — used only when --full is passed, to bypass the
# hardcoded 150-sample subset in get_subset_indices() (N_SUBSET in
# src/selector.py is fixed at 150, so it cannot return more than that
# regardless of dataset — --full computes the true full index range here).
DATASET_SIZES = {
    "commonvoice":      680,
    "english_dialects": 2543,
    "edacc":            198,
    "shetland":         100,
}

WHISPERX_MODEL_SIZE = "large-v3"

NEW_OUTPUT_DIR = "writeup_results/benchmarks/main"
OLD_OUTPUT_DIR = "results/benchmarks/main"


def get_device():
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def run_dataset(dataset_name: str, max_samples: int = None, rerun: bool = False,
                 full: bool = False):
    print(f"\n── WhisperX transcription only (PHASE 1): {dataset_name} ──")
    print(f"  (no severity judging here - run add_severity_to_existing.py after)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    if full:
        new_output_path = f"{NEW_OUTPUT_DIR}/whisperx_{dataset_name}_{timestamp}.json"
        old_output_path = f"{OLD_OUTPUT_DIR}/whisperx_{dataset_name}_{timestamp}.json"
        # full runs always write fresh timestamped files — never overwrites
        # existing sub150 files OR previous full runs
    else:
        new_output_path = os.path.join(NEW_OUTPUT_DIR, f"whisperx_{dataset_name}_sub150.json")
        old_output_path = os.path.join(SUBSETS_DIR, f"whisperx_{dataset_name}_sub150.json")

        if os.path.exists(old_output_path) and not rerun:
            with open(old_output_path) as f:
                existing = json.load(f)
            n_done = len(existing.get("samples", []))
            print(f"  Already exists with {n_done} samples. Use --rerun to redo.")
            return

    if full:
        indices = list(range(DATASET_SIZES[dataset_name]))
        print(f"  FULL dataset mode: {len(indices)} samples (not the 150-subset)")
    else:
        indices = get_subset_indices(dataset_name)
    if max_samples:
        indices = indices[:max_samples]
    indices_set = set(indices)

    print(f"  {len(indices)} samples to transcribe")

    # load WhisperX only — no Ollama client, no Phi-4, nothing else
    # competing for memory during this phase
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
    print("  Model ready.")

    # load dataset and iterate
    dataset_cls = DATASETS[dataset_name]
    dataset     = dataset_cls()

    results    = {}  # sample_index -> result
    start_time = time.time()

    def save_progress():
        ordered = [results[i] for i in indices if i in results]
        payload = {
            "model":   "whisperx",
            "dataset": dataset_name,
            "phase":   "transcription_only - severity not yet judged",
            "samples": ordered,
        }
        with open(new_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(old_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

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
                "severity":     None,
                "segments":     [],
                "skipped":      True,
                "skip_reason":  "ignore_time_segment",
            }
            continue

        # skip refs that are entirely bracketed annotation tags (e.g. "<OVERLAP>")
        # — no real reference content exists to score the hypothesis against.
        if is_tag_only(ref):
            results[global_idx] = {
                "sample_index": global_idx,
                "ref":          ref,
                "hyp":          None,
                "sample_WER":   None,
                "severity":     None,
                "segments":     [],
                "skipped":      True,
                "skip_reason":  "tag_only_reference",
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

            results[global_idx] = {
                "sample_index": global_idx,
                "ref":          ref,
                "hyp":          hyp,
                "sample_WER":   sample_wer,
                "severity":     None,   # filled in later by phase 2
                "segments":     word_segments,
            }

            if (len(results)) % 10 == 0:
                save_progress()

            print(f"  [{len(results)}/{len(indices)}] {global_idx}: "
                  f"WER={sample_wer*100:.1f}%  "
                  f"({time.time()-start_time:.0f}s elapsed)")

        except Exception as e:
            print(f"  ERROR sample {global_idx}: {e}")
            results[global_idx] = {
                "sample_index": global_idx,
                "ref":          ref,
                "hyp":          "",
                "sample_WER":   None,
                "severity":     None,
                "segments":     [],
                "error":        str(e),
            }

        # stop early if we have all needed indices
        if len(results) >= len(indices):
            break

    # build ordered output
    ordered = [results[i] for i in indices if i in results]
    valid   = [s for s in ordered if s.get("sample_WER") is not None]
    n_skipped = sum(1 for s in ordered if s.get("skipped"))

    corpus_wer_val = wer(
        [normalise(s["ref"]) for s in valid],
        [normalise(s["hyp"]) for s in valid],
    ) if valid else None

    output = {
        "model":          "whisperx",
        "dataset":        dataset_name,
        "model_size":     WHISPERX_MODEL_SIZE,
        "phase":          "transcription_only - severity not yet judged",
        "corpus_wer":     corpus_wer_val,
        "num_samples":    len(valid),
        "num_skipped":    n_skipped,
        "subset_indices": indices,
        "samples":        ordered,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)}, skipped={n_skipped})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2 to add severity scores:")
    print(f"  python rerunning/add_severity_to_existing.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default="commonvoice",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun",       action="store_true")
    parser.add_argument("--full",        action="store_true",
                        help="Run on the FULL dataset (bypasses the 150-sample "
                             "subset). Writes fresh timestamped files to both "
                             "writeup_results/ and results/ — never overwrites "
                             "existing sub150 files or previous full runs.")
    args = parser.parse_args()

    run_dataset(args.dataset,
                max_samples=args.max_samples,
                rerun=args.rerun,
                full=args.full)


if __name__ == "__main__":
    main()