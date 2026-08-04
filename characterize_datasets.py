"""
characterize_datasets.py

Verifies actual audio duration by reading real audio files directly
(librosa), rather than relying on any prior estimate. Also produces
dataset characterization stats for the datasets chapter: word/char
count distributions, filler-only utterance rate, vocabulary size/
type-token ratio, and example transcripts (shortest/median/longest).

Reuses your existing dataset loader classes (src/datasets.py) - same
ones used throughout the pipeline - so counts reflect your ACTUAL
subsets, not the full public corpora.

Usage:
    python characterize_datasets.py --datasets commonvoice english_dialects edacc
    python characterize_datasets.py --datasets commonvoice --examples 5
"""

import argparse
import statistics
from collections import Counter

import librosa

from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc

DATASETS = {
    "commonvoice": CommonVoiceScots,
    "english_dialects": EnglishDialectsScots,
    "edacc": EdAcc,
}

FILLER_WORDS = {
    "um", "uh", "umm", "uhh", "erm", "er", "mm", "mmm", "hmm", "hm",
    "mhm", "yeah", "like", "so", "you know", "i mean",
}


def characterize(dataset_key: str, n_examples: int = 3):
    dataset_cls = DATASETS[dataset_key]
    dataset = dataset_cls()

    print(f"\n{'='*70}")
    print(f"  {dataset_key.upper()}")
    print(f"{'='*70}")

    durations = []
    word_counts = []
    char_counts = []
    all_words = []
    filler_only_count = 0
    n_samples = 0
    transcripts_with_length = []

    for sample in dataset.load():
        n_samples += 1
        audio = sample.audio
        sr = sample.sample_rate
        duration_sec = len(audio) / sr
        durations.append(duration_sec)

        text = sample.label.strip()
        words = text.lower().split()
        word_counts.append(len(words))
        char_counts.append(len(text))
        all_words.extend(words)

        non_filler = [w.strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"") not in FILLER_WORDS]
        if words and not non_filler:
            filler_only_count += 1

        transcripts_with_length.append((len(words), text))

    total_hours = sum(durations) / 3600
    mean_dur = statistics.mean(durations) if durations else 0
    median_dur = statistics.median(durations) if durations else 0

    print(f"\n-- Verified duration (read directly from audio) --")
    print(f"  N samples:       {n_samples}")
    print(f"  Total duration:  {total_hours:.3f} hours ({sum(durations):.1f} sec)")
    print(f"  Mean duration:   {mean_dur:.2f} sec")
    print(f"  Median duration: {median_dur:.2f} sec")
    print(f"  Min/Max:         {min(durations):.2f}s / {max(durations):.2f}s" if durations else "  (no data)")

    print(f"\n-- Transcript characteristics --")
    if word_counts:
        print(f"  Word count: mean={statistics.mean(word_counts):.1f}  "
              f"median={statistics.median(word_counts):.1f}  "
              f"min={min(word_counts)}  max={max(word_counts)}  "
              f"stdev={statistics.stdev(word_counts):.1f}" if len(word_counts) > 1 else "")
        print(f"  Char count: mean={statistics.mean(char_counts):.1f}  "
              f"median={statistics.median(char_counts):.1f}")
        print(f"  Filler-only utterances: {filler_only_count}/{n_samples} "
              f"({filler_only_count/n_samples*100:.1f}%)")

    vocab = set(all_words)
    ttr = len(vocab) / len(all_words) if all_words else 0
    print(f"\n-- Vocabulary --")
    print(f"  Total word tokens: {len(all_words)}")
    print(f"  Unique words (vocab size): {len(vocab)}")
    print(f"  Type-token ratio: {ttr:.4f}")

    transcripts_with_length.sort(key=lambda x: x[0])
    print(f"\n-- Example transcripts (shortest / median / longest by word count) --")
    if transcripts_with_length:
        print(f"  Shortest: \"{transcripts_with_length[0][1]}\"")
        mid_idx = len(transcripts_with_length) // 2
        print(f"  Median:   \"{transcripts_with_length[mid_idx][1]}\"")
        print(f"  Longest:  \"{transcripts_with_length[-1][1][:150]}{'...' if len(transcripts_with_length[-1][1]) > 150 else ''}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    for dataset_key in args.datasets:
        characterize(dataset_key, n_examples=args.examples)


if __name__ == "__main__":
    main()