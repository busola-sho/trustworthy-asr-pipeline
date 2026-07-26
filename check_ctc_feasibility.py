"""
check_ctc_feasibility.py

Scans a manifest file (train.jsonl/val.jsonl from prepare_manifest.py) and
checks, per sample, whether CTC alignment is even POSSIBLE given the
audio's encoded sequence length vs. its transcript's character length.

wav2vec2's CNN feature encoder downsamples ~320x, so encoded frame count
is roughly duration_seconds * 16000 / 320 (~50 frames/sec). Standard CTC
needs encoded_frames >= 2*label_length - 1 (to allow for blank tokens
between repeated characters) - samples below this threshold get zeroed
out by ctc_zero_infinity=True rather than contributing real loss, which
is invisible in training logs (just looks like part of an averaged loss)
but adds up if it affects a large fraction of your data.

Usage:
    python check_ctc_feasibility.py data/finetune/manifests/train.jsonl
"""

import json
import sys

import soundfile as sf

DOWNSAMPLE_FACTOR = 320
SAMPLE_RATE = 16000


def encoded_frame_count(duration_sec: float) -> int:
    return int(duration_sec * SAMPLE_RATE / DOWNSAMPLE_FACTOR)


def main():
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "data/finetune/manifests/train.jsonl"

    total = 0
    infeasible = 0
    marginal = 0   # feasible but with little headroom (< 1.5x the minimum)
    durations = []

    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            total += 1

            try:
                info = sf.info(entry["audio_path"])
                duration = info.frames / info.samplerate
            except Exception as e:
                print(f"  Could not read {entry['audio_path']}: {e}")
                continue

            durations.append(duration)
            text = entry["text"].replace(" ", "")  # CTC label length ~ character count
            label_len = len(text)
            frames = encoded_frame_count(duration)
            min_required = 2 * label_len - 1 if label_len > 0 else 1

            if frames < min_required:
                infeasible += 1
                if infeasible <= 10:
                    print(f"  INFEASIBLE: {entry['audio_path']} - {duration:.2f}s -> "
                          f"{frames} frames, needs >= {min_required} for {label_len} chars")
            elif frames < min_required * 1.5:
                marginal += 1

    durations.sort()
    print(f"\nTotal samples: {total}")
    print(f"INFEASIBLE (CTC impossible, silently zeroed by ctc_zero_infinity): {infeasible} "
          f"({infeasible/total*100:.1f}%)")
    print(f"MARGINAL (feasible but little headroom): {marginal} ({marginal/total*100:.1f}%)")
    if durations:
        print(f"\nDuration stats: min={durations[0]:.2f}s  "
              f"median={durations[len(durations)//2]:.2f}s  max={durations[-1]:.2f}s")
        short_count = sum(1 for d in durations if d < 0.5)
        print(f"Samples under 0.5s: {short_count} ({short_count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
