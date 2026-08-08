"""
sat_postprocess.py

Locked SaT post-processing protocol (final revision, round 3):
  1. Keep SaT boundaries by default.
  2. Rule A: split ALL clear punctuation-plus-capital boundaries,
     protected by an abbreviation exception list (Mr./Dr./St./etc.)
     AND an initial/acronym guard (catches multi-period cases like
     "U.S." or "A. Smith" the plain word-list can't).
  3. Rule B: merge a reporting clause with its quotation.
  4. Short-fragment flagging only, never auto-merged.
  5. Content-preservation invariant enforced after every run.
  6. BOTH Rule A and Rule B changes are logged, each run gets a fresh
     log file (mode "w" with a run header) rather than appending across
     runs, which would mix entries from different executions.

VALIDATION (final design): the unit of evaluation is the INDIVIDUAL
BOUNDARY CHANGE, not the whole transcript - a single transcript-level
label was too coarse (could hide one correct + one erroneous change in
the same transcript). For each of a stratified, matched sample of
transcripts:
  - every actual Rule A split is labeled individually:
    [C]orrect / [N]eutral(unnecessary) / [E]rroneous
  - every actual Rule B merge is labeled individually, same scale
  - remaining SaT boundaries that NEITHER rule touched are separately
    flagged for missed-split / incorrect-split review
This gives per-rule precision, not just one aggregate number.

Locked final protocol:
  1. Develop/validate rules on DEV transcripts only.
  2. Stratified, matched, change-level validation.
  3. Inspect the audit log for Rule A/B false positives.
  4. Freeze the rules.
  5. Apply the frozen pipeline UNCHANGED to test and Shetland.

Usage:
    python sat_postprocess.py --dataset commonvoice --n-samples 5
    python sat_postprocess.py --validate --n-per-dataset 17
"""

import json
import re
import random
import argparse
from datetime import datetime

from wtpsplit import SaT

DATASETS = ["commonvoice", "edacc", "english_dialects"]
RULE_LOG_PATH = "rule_ab_splits_log.txt"

SHORT_FRAGMENT_MAX_WORDS = 4
SENTENCE_FINAL_PUNCT = re.compile(r'[.!?]["\')]?\s*$')

MID_SEGMENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c\u2018])')
OPENING_QUOTES = ('"', "'", "\u201c", "\u2018")

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "rev", "st", "jr", "sr",
    "vs", "etc", "e.g", "i.e", "mt", "ft", "no", "capt", "sgt",
    "col", "gen", "lt", "cpl", "maj",
}

# catches initials/acronyms like "A." or multi-period abbreviations
# like "U.S." that the plain word-list can't - e.g. "A. Smith" or
# "the U.S. government"
INITIAL_OR_ACRONYM = re.compile(r'(?:\b[A-Z]\.)+\s*$')


def load_5model_transcripts(dataset, split="dev"):
    path = f"writeup_results/grid/unanchored_fusion_naive_5model/unanchored_fusion_naive_5model_{dataset}_gemma4_{split}.json"
    data = json.load(open(path))
    samples = data.get("samples", [])
    return [s for s in samples if s.get("hyp") and not s.get("skipped") and not s.get("error")]


def _is_abbreviation_before(text_before_period):
    if INITIAL_OR_ACRONYM.search(text_before_period):
        return True
    match = re.search(r'(\w+)\.\s*$', text_before_period)
    if not match:
        return False
    return match.group(1).lower() in ABBREVIATIONS


def rule_a_split_merged_sentences(segments):
    """Returns (result_segments, split_events) - split_events is a list
    of (original_merged_text, [resulting_pieces]) for EVERY actual
    split made, so each can be individually labeled/logged."""
    result = []
    split_events = []
    for segment in segments:
        stripped = segment.strip()
        pieces = []
        last_end = 0
        for match in MID_SEGMENT_BOUNDARY.finditer(stripped):
            candidate_before = stripped[:match.start()]
            if _is_abbreviation_before(candidate_before):
                continue
            pieces.append(stripped[last_end:match.start()].strip())
            last_end = match.start()
        pieces.append(stripped[last_end:].strip())
        pieces = [p for p in pieces if p]

        if len(pieces) > 1:
            split_events.append((stripped, pieces))

        result.extend(p + " " for p in pieces)
    return result, split_events


def rule_b_merge_reporting_quotation(segments):
    """Returns (result_segments, merge_events) - merge_events is a list
    of (before_seg, after_seg, merged_result) for EVERY actual merge."""
    if not segments:
        return segments, []
    result = [segments[0]]
    merge_events = []
    for seg in segments[1:]:
        prev = result[-1]
        prev_ends_open = prev.rstrip().endswith((",", ":"))
        curr_starts_quote = seg.lstrip().startswith(OPENING_QUOTES)
        if prev_ends_open and curr_starts_quote:
            merged = prev.rstrip() + " " + seg.strip() + " "
            merge_events.append((prev, seg, merged))
            result[-1] = merged
        else:
            result.append(seg)
    return result, merge_events


def flag_short_fragments(segments):
    flagged = []
    for i, seg in enumerate(segments):
        word_count = len(seg.split())
        has_final_punct = bool(SENTENCE_FINAL_PUNCT.search(seg.strip()))
        if word_count < SHORT_FRAGMENT_MAX_WORDS and not has_final_punct:
            flagged.append((i, seg))
    return flagged


def normalise_for_boundary_check(segments):
    return re.sub(r"\s+", " ", "".join(segments)).strip()


def apply_postprocessing(raw_segments):
    """Returns (final_segments, split_events, merge_events)."""
    after_a, split_events = rule_a_split_merged_sentences(raw_segments)
    after_b, merge_events = rule_b_merge_reporting_quotation(after_a)

    raw_text = normalise_for_boundary_check(raw_segments)
    processed_text = normalise_for_boundary_check(after_b)
    if raw_text != processed_text:
        raise ValueError(
            "Post-processing altered transcript content - this must never happen. "
            f"RAW: {raw_text!r}\nPROCESSED: {processed_text!r}"
        )
    return after_b, split_events, merge_events


def log_events(log_file, dataset, idx, split_events, merge_events):
    for original, pieces in split_events:
        log_file.write(f"[{dataset} #{idx}] RULE A SPLIT:\n")
        log_file.write(f"    BEFORE: {original!r}\n")
        for p in pieces:
            log_file.write(f"    AFTER:  {p!r}\n")
        log_file.write("\n")
    for before, after_seg, merged in merge_events:
        log_file.write(f"[{dataset} #{idx}] RULE B MERGE:\n")
        log_file.write(f"    BEFORE: {before!r}\n")
        log_file.write(f"            {after_seg!r}\n")
        log_file.write(f"    AFTER:  {merged!r}\n")
        log_file.write("\n")


def inspect(args):
    print("Loading SaT model...")
    sat = SaT("sat-3l")

    samples = load_5model_transcripts(args.dataset, args.split)
    print(f"Loaded {len(samples)} valid transcripts from {args.dataset} ({args.split})\n")

    rng = random.Random(42)
    selected = rng.sample(samples, min(args.n_samples, len(samples)))

    with open(RULE_LOG_PATH, "w") as log_file:
        log_file.write(f"Rule A/B audit log - run started {datetime.now().isoformat()}\n")
        log_file.write(f"Mode: inspect --dataset {args.dataset} --n-samples {args.n_samples}\n\n")

        for i, s in enumerate(selected):
            hyp = s["hyp"]
            raw_segments = sat.split(hyp)
            processed, split_events, merge_events = apply_postprocessing(raw_segments)
            flagged = flag_short_fragments(processed)
            log_events(log_file, args.dataset, s.get("dataset_index"), split_events, merge_events)

            print(f"{'='*90}")
            print(f"SAMPLE {i} (dataset_index={s.get('dataset_index')})")
            print(f"{'='*90}")
            print(f"RAW SaT ({len(raw_segments)} segments):")
            for j, seg in enumerate(raw_segments):
                print(f"  [{j}] {seg!r}")
            print(f"\nAFTER RULES A+B ({len(processed)} segments, "
                  f"{len(split_events)} rule-A splits, {len(merge_events)} rule-B merges):")
            for j, seg in enumerate(processed):
                flag_marker = "  <- FLAGGED (short fragment)" if any(fi == j for fi, _ in flagged) else ""
                print(f"  [{j}] {seg!r}{flag_marker}")
            print()

    print(f"Rule A/B changes logged to: {RULE_LOG_PATH}")


def validate(args):
    """Stratified + matched + CHANGE-LEVEL validation. For each
    dataset, samples a fixed number of transcripts, then for each one
    individually labels every actual Rule A split and Rule B merge -
    not one coarse label per transcript."""
    print("Loading SaT model...")
    sat = SaT("sat-3l")

    print(f"\n{'='*90}")
    print(f"CHANGE-LEVEL VALIDATION - {args.n_per_dataset} transcripts per dataset "
          f"({args.n_per_dataset * len(DATASETS)} total)")
    print(f"For each Rule A split / Rule B merge, label: [C]orrect / [N]eutral / [E]rroneous")
    print(f"Also separately note any remaining SaT boundaries that look like missed or")
    print(f"incorrect splits (untouched by either rule).")
    print(f"{'='*90}")

    n_rule_a_total = 0
    n_rule_b_total = 0

    with open(RULE_LOG_PATH, "w") as log_file:
        log_file.write(f"Rule A/B audit log - validation run started {datetime.now().isoformat()}\n")
        log_file.write(f"Mode: validate --n-per-dataset {args.n_per_dataset}\n\n")

        for dataset in DATASETS:
            samples = load_5model_transcripts(dataset, "dev")
            rng = random.Random(42)
            selected = rng.sample(samples, min(args.n_per_dataset, len(samples)))

            for s in selected:
                idx = s.get("dataset_index")
                raw_segments = sat.split(s["hyp"])
                processed, split_events, merge_events = apply_postprocessing(raw_segments)
                log_events(log_file, dataset, idx, split_events, merge_events)
                n_rule_a_total += len(split_events)
                n_rule_b_total += len(merge_events)

                if not split_events and not merge_events:
                    continue  # nothing changed for this transcript - skip presenting it

                print(f"\n{'-'*90}")
                print(f"TRANSCRIPT ({dataset}, sample {idx})")
                print(f"{'-'*90}")

                for k, (original, pieces) in enumerate(split_events):
                    print(f"\n  RULE A SPLIT {k+1}:")
                    print(f"    BEFORE: {original!r}")
                    for p in pieces:
                        print(f"    AFTER:  {p!r}")
                    print(f"    LABEL: [ ]")

                for k, (before, after_seg, merged) in enumerate(merge_events):
                    print(f"\n  RULE B MERGE {k+1}:")
                    print(f"    BEFORE: {before!r}")
                    print(f"            {after_seg!r}")
                    print(f"    AFTER:  {merged!r}")
                    print(f"    LABEL: [ ]")

                print(f"\n  FULL PROCESSED TRANSCRIPT (for context, check remaining boundaries):")
                for j, seg in enumerate(processed):
                    print(f"    [{j}] {seg!r}")
                print(f"  REMAINING ISSUES (missed/incorrect splits not touched by A/B): ____")

    print(f"\n{'='*90}")
    print(f"Total Rule A splits fired: {n_rule_a_total}")
    print(f"Total Rule B merges fired: {n_rule_b_total}")
    print(f"After labeling, compute per-rule precision (C / (C+N+E)) and note any")
    print(f"remaining missed/incorrect SaT boundaries separately.")
    print(f"Full audit log: {RULE_LOG_PATH}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice", choices=DATASETS)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--n-per-dataset", type=int, default=17)
    args = parser.parse_args()

    if args.validate:
        validate(args)
    else:
        inspect(args)


if __name__ == "__main__":
    main()
