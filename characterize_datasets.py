"""
characterize_datasets.py

Objective, quantitative characterization of each dataset used in the
dissertation - designed to give a reader/reviewer a concrete, evidence-based
sense of what each dataset actually contains, rather than a bare sample
count. Computes, per dataset:

  - Utterance length distribution (words and characters): mean, median,
    std, min, max, plus the proportion of utterances at or below a short
    threshold (default 3 words) - a concrete, citable number for claims
    like "EdAcc contains many very short utterances."
  - Audio duration distribution (from WhisperX's word-level timestamps).
  - Vocabulary size and type-token ratio (lexical diversity).
  - Proportion of utterances that are ENTIRELY filler/backchannel words
    (e.g. "mmm", "uh-huh") - directly quantifies short/low-content
    utterance prevalence, rather than just asserting it.
  - A handful of ACTUAL example transcripts (shortest, a few representative
    mid-length ones, longest) so a reader gets a qualitative feel for the
    data alongside the numbers.

Then prints a final side-by-side comparison table across all datasets, so
patterns (e.g. "EdAcc's short-utterance proportion is far higher than the
other three") are visible at a glance, not buried in per-dataset reports.

Deliberately descriptive, not evaluative - this reports what IS in each
dataset (lengths, vocabulary, examples) without characterizing any
technique's performance on it. Keeping description and evaluation
separate is what makes a "this dataset is harder to model" claim
elsewhere in the dissertation land as rigorous rather than as an
after-the-fact excuse.

Usage:
    python characterize_datasets.py
    python characterize_datasets.py --datasets commonvoice edacc
    python characterize_datasets.py --short-threshold 3
"""

import argparse
import statistics
import random
from collections import defaultdict

from src.selector import find_canonical_file, load_samples
from src.text_normalise import normalise

DATASETS = ["commonvoice", "english_dialects", "edacc", "shetland"]

# Preferred model to pull reference transcripts + timestamps from, in
# priority order - WhisperX first since it has word-level start/end
# timestamps needed for audio duration; falls back to whichever model
# actually has a file for a given dataset.
MODEL_PRIORITY = ["whisperx", "qwen", "parakeet", "wav2vec2"]

# Conservative filler/backchannel word list - an utterance is flagged as
# "filler-only" only if EVERY word in it is in this set, after
# normalisation. Deliberately narrow (real content words are never
# included) so this doesn't overstate filler prevalence.
FILLER_WORDS = {
    "mm", "mmm", "mhm", "mmhm", "uh", "um", "erm", "hmm", "huh",
    "uhhuh", "mhmm", "aye", "yeah", "yep", "nah", "no", "yes", "ok", "okay",
    "right", "so", "well", "oh", "ah",
}


def load_dataset_samples(dataset: str):
    """Loads reference transcripts + timestamps from whichever model has a
    file for this dataset, preferring WhisperX for its word-level
    timestamps. Returns (samples, model_used)."""
    for model in MODEL_PRIORITY:
        try:
            path = find_canonical_file(model, dataset)
            samples = load_samples(path)
            if samples:
                return samples, model
        except FileNotFoundError:
            continue
    return [], None


def get_duration(sample: dict):
    """Audio duration in seconds from word-level segment timestamps, if
    available (first word's start to last word's end)."""
    segments = sample.get("segments")
    if not segments:
        return None
    starts = [s.get("start") for s in segments if s.get("start") is not None]
    ends = [s.get("end") for s in segments if s.get("end") is not None]
    if not starts or not ends:
        return None
    return max(ends) - min(starts)


def is_filler_only(words: list) -> bool:
    if not words:
        return False
    return all(w.lower() in FILLER_WORDS for w in words)


def characterize(dataset: str, short_threshold: int, n_examples: int):
    samples, model_used = load_dataset_samples(dataset)
    if not samples:
        return None

    word_counts, char_counts, durations = [], [], []
    filler_only_count = 0
    valid_refs = []
    vocab = set()
    total_words = 0

    for s in samples:
        ref = s.get("ref")
        if not ref or s.get("skipped"):
            continue
        ref = ref.strip()
        if not ref:
            continue

        norm = normalise(ref)
        words = norm.split()
        if not words:
            continue

        word_counts.append(len(words))
        char_counts.append(len(ref))
        total_words += len(words)
        vocab.update(words)

        if is_filler_only(words):
            filler_only_count += 1

        dur = get_duration(s)
        if dur is not None and dur > 0:
            durations.append(dur)

        valid_refs.append(ref)

    if not word_counts:
        return None

    n = len(word_counts)
    short_count = sum(1 for w in word_counts if w <= short_threshold)

    # example transcripts: shortest, longest, and a few spread across the
    # middle of the length distribution (not just random - deliberately
    # picked from evenly-spaced length percentiles so the examples are
    # representative of the actual range, not clustered)
    sorted_by_len = sorted(zip(word_counts, valid_refs), key=lambda x: x[0])
    shortest = sorted_by_len[:n_examples]
    longest = sorted_by_len[-n_examples:]
    mid_indices = [int(n * p) for p in [0.25, 0.5, 0.75]] if n >= 4 else []
    mid_examples = [sorted_by_len[i] for i in mid_indices if i < n]

    stats = {
        "dataset": dataset,
        "model_used_for_refs": model_used,
        "n_utterances": n,
        "word_count_mean": statistics.mean(word_counts),
        "word_count_median": statistics.median(word_counts),
        "word_count_std": statistics.stdev(word_counts) if n > 1 else 0.0,
        "word_count_min": min(word_counts),
        "word_count_max": max(word_counts),
        "short_threshold": short_threshold,
        "short_count": short_count,
        "short_pct": short_count / n * 100,
        "char_count_mean": statistics.mean(char_counts),
        "vocab_size": len(vocab),
        "total_words": total_words,
        "type_token_ratio": len(vocab) / total_words if total_words else None,
        "filler_only_count": filler_only_count,
        "filler_only_pct": filler_only_count / n * 100,
        "duration_mean": statistics.mean(durations) if durations else None,
        "duration_median": statistics.median(durations) if durations else None,
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
        "duration_n": len(durations),
        "examples_shortest": shortest,
        "examples_mid": mid_examples,
        "examples_longest": longest,
    }
    return stats


def print_dataset_report(stats: dict):
    d = stats["dataset"]
    print(f"\n{'=' * 70}")
    print(f"  {d.upper()}  (reference transcripts from: {stats['model_used_for_refs']})")
    print(f"{'=' * 70}")
    print(f"  N utterances (scoreable):      {stats['n_utterances']}")
    print(f"  Word count:  mean={stats['word_count_mean']:.1f}  median={stats['word_count_median']:.1f}  "
          f"std={stats['word_count_std']:.1f}  min={stats['word_count_min']}  max={stats['word_count_max']}")
    print(f"  Utterances <= {stats['short_threshold']} words: {stats['short_count']} "
          f"({stats['short_pct']:.1f}%)")
    print(f"  Mean character count: {stats['char_count_mean']:.1f}")
    print(f"  Vocabulary size: {stats['vocab_size']} unique words "
          f"(type-token ratio: {stats['type_token_ratio']:.4f})" if stats['type_token_ratio'] else "")
    print(f"  Filler-only utterances (e.g. \"mm\", \"uh-huh\"): {stats['filler_only_count']} "
          f"({stats['filler_only_pct']:.1f}%)")
    if stats["duration_n"] > 0:
        print(f"  Audio duration (n={stats['duration_n']}): mean={stats['duration_mean']:.1f}s  "
              f"median={stats['duration_median']:.1f}s  min={stats['duration_min']:.1f}s  "
              f"max={stats['duration_max']:.1f}s")
    else:
        print(f"  Audio duration: not available (no word-level timestamps in this file)")

    print(f"\n  Example transcripts (shortest):")
    for wc, ref in stats["examples_shortest"]:
        print(f"    [{wc:>3} words] \"{ref}\"")
    if stats["examples_mid"]:
        print(f"\n  Example transcripts (25th/50th/75th percentile length):")
        for wc, ref in stats["examples_mid"]:
            ref_display = ref if len(ref) <= 120 else ref[:117] + "..."
            print(f"    [{wc:>3} words] \"{ref_display}\"")
    print(f"\n  Example transcripts (longest):")
    for wc, ref in stats["examples_longest"]:
        ref_display = ref if len(ref) <= 120 else ref[:117] + "..."
        print(f"    [{wc:>3} words] \"{ref_display}\"")


def print_comparison_table(all_stats: list):
    print(f"\n{'=' * 70}")
    print(f"  CROSS-DATASET COMPARISON")
    print(f"{'=' * 70}")

    headers = ["Dataset", "N", "Mean words", "Median words", f"% short", "Filler-only %", "Mean dur (s)"]
    col_widths = [max(len(h), 12) for h in headers]

    def fmt_row(cells):
        return "  ".join(f"{str(c):<{w}}" for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for s in all_stats:
        dur_str = f"{s['duration_mean']:.1f}" if s["duration_mean"] is not None else "-"
        print(fmt_row([
            s["dataset"], s["n_utterances"],
            f"{s['word_count_mean']:.1f}", f"{s['word_count_median']:.1f}",
            f"{s['short_pct']:.1f}%", f"{s['filler_only_pct']:.1f}%", dur_str,
        ]))

    print(f"\n  NOTE: 'short' = utterances at or below {all_stats[0]['short_threshold']} words. "
          f"'Filler-only' = utterances consisting ENTIRELY of backchannel/filler words "
          f"(e.g. \"mm\", \"uh-huh\") - a conservative, narrow word list, so this likely "
          f"UNDERSTATES rather than overstates low-content utterance prevalence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--short-threshold", type=int, default=3,
                        help="Utterances at or below this many words are counted as 'short' (default: 3)")
    parser.add_argument("--n-examples", type=int, default=3,
                        help="How many example transcripts to show per length category (default: 3)")
    args = parser.parse_args()

    random.seed(42)  # not currently used for randomness, but reserved for reproducibility if added later

    all_stats = []
    for dataset in args.datasets:
        stats = characterize(dataset, args.short_threshold, args.n_examples)
        if stats is None:
            print(f"\nSkipping {dataset} - no usable reference data found")
            continue
        print_dataset_report(stats)
        all_stats.append(stats)

    if all_stats:
        print_comparison_table(all_stats)


if __name__ == "__main__":
    main()