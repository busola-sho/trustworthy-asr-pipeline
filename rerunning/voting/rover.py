"""
rerunning/voting/rover.py

PHASE 1 of 2: ROVER (Recognizer Output Voting Error Reduction), following
Fiscus (1997) - "A Post-Processing System to Yield Reduced Word Error
Rates: Recognizer Output Voting Error Reduction (ROVER)".

This is a SIMPLE, STRAIGHT implementation of the original algorithm:
  1. Build a Word Transition Network (WTN) by incrementally aligning each
     system's hypothesis into a growing composite alignment, one system
     at a time (pairwise dynamic-programming/Levenshtein alignment against
     the current WTN's majority-word representative sequence).
  2. At each aligned position ("column"), take a majority vote among the
     words proposed by each system (a system can also contribute NULL,
     meaning "no word here" - a deletion).
  3. Ties are broken using each system's average word confidence for its
     proposed word at that column (matches Fiscus's own score-based
     tiebreak, now that all 4 models can produce confidence scores).
  4. The final transcript is the concatenation of winning (non-NULL)
     words across all columns, in order.

NO LLM is used anywhere in this phase - no selector, no semantic auditor,
no dialect pass. Just alignment + voting, exactly per the paper. Severity
judging is a separate PHASE 2, same as every other technique:
    python rerunning/add_severity_to_existing.py --files <output file from this script>

Writes to BOTH locations:
  - writeup_results/voting/rover/rover_{dataset}_{split}.json      (new)
  - results/combinations_v2judge/rover_whisperx/rover_{dataset}_{split}.json (old, kept for continuity)

Same --split {dev,test,full} pattern as the ensemble scripts (replaces
--full; defaults to "dev" for iteration).

Usage:
    python rerunning/voting/rover.py --dataset commonvoice --split dev
    python rerunning/voting/rover.py --dataset edacc --split full
"""

import json
import os
import argparse
from jiwer import wer

from src.judge import normalise, is_tag_only
from src.selector import find_canonical_file, load_samples
from src.splits import get_indices_for_split

NEW_OUTPUT_DIR = "writeup_results/voting/rover"
OLD_OUTPUT_DIR = "results/combinations_v2judge/rover_whisperx"
DATASETS       = ["commonvoice", "english_dialects", "edacc", "shetland"]
ASR_MODELS     = ["qwen", "whisperx", "parakeet", "wav2vec2"]   # whisper -> whisperx


# ── Pairwise alignment (Levenshtein-style DP, per Fiscus's DPA module) ─────────

def align_sequences(seq_a: list, seq_b: list) -> list:
    """
    Standard edit-distance alignment between two word sequences.
    Returns a list of (word_a_or_None, word_b_or_None) pairs - None marks
    a gap (insertion/deletion) on that side. Match cost 0, substitution
    cost 1, insertion/deletion cost 1. Case-insensitive comparison.
    """
    n, m = len(seq_a), len(seq_b)
    # dp[i][j] = min cost aligning seq_a[:i] with seq_b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = 0 if seq_a[i - 1].lower() == seq_b[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + match_cost,  # substitution or match
                dp[i - 1][j] + 1,               # deletion (a has extra word)
                dp[i][j - 1] + 1,               # insertion (b has extra word)
            )

    # backtrace
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_cost = 0 if seq_a[i - 1].lower() == seq_b[j - 1].lower() else 1
            if dp[i][j] == dp[i - 1][j - 1] + match_cost:
                aligned.append((seq_a[i - 1], seq_b[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((seq_a[i - 1], None))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned.append((None, seq_b[j - 1]))
            j -= 1
            continue
        break

    aligned.reverse()
    return aligned


# ── Word Transition Network construction ───────────────────────────────────────

def representative_words(wtn: list, model_order: list) -> list:
    """
    Builds the sequence used to align the NEXT system against - one word
    per existing column, using the current majority word so far (or the
    first available system's word if no majority yet). This is what the
    new hypothesis gets aligned against, per Fiscus's incremental merge.
    """
    reps = []
    for column in wtn:
        words = [column.get(m) for m in model_order if column.get(m) is not None]
        if not words:
            reps.append("")
            continue
        # majority so far, ties broken by first-seen order
        counts = {}
        for w in words:
            counts[w.lower()] = counts.get(w.lower(), 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        # recover original casing from the first matching word
        reps.append(next(w for w in words if w.lower() == best))
    return reps


def build_wtn(hyps: dict, model_order: list) -> list:
    """
    Incrementally builds the Word Transition Network by aligning each
    system's hypothesis into the growing composite, one at a time.
    Returns a list of columns, each a dict {model_name: word_or_None}.
    """
    first_model = model_order[0]
    first_words = hyps[first_model].split()
    wtn = [{first_model: w} for w in first_words]

    for model in model_order[1:]:
        new_words = hyps[model].split()
        reps = representative_words(wtn, model_order)
        alignment = align_sequences(reps, new_words)

        merged = []
        wtn_idx = 0
        for rep_word, new_word in alignment:
            if rep_word is not None and new_word is not None:
                # aligned to an existing column - extend it
                column = dict(wtn[wtn_idx])
                column[model] = new_word
                merged.append(column)
                wtn_idx += 1
            elif rep_word is not None and new_word is None:
                # existing column has no match in the new hypothesis - deletion for this model
                column = dict(wtn[wtn_idx])
                column[model] = None
                merged.append(column)
                wtn_idx += 1
            else:
                # new hypothesis has an extra word not in the existing WTN - new column,
                # None for every system that came before this one
                column = {m: None for m in model_order if m != model}
                column[model] = new_word
                merged.append(column)

        wtn = merged

    # fill in any missing models per column with None (systems that never
    # got a chance to vote at that column)
    for column in wtn:
        for m in model_order:
            if m not in column:
                column[m] = None

    return wtn


# ── Voting ──────────────────────────────────────────────────────────────────────

def get_word_confidence(model: str, word: str, segments: list) -> float:
    """Looks up the confidence score for a specific word from a model's
    segments list. Falls back to 0.5 if not found (shouldn't normally
    happen, but keeps voting well-defined if segment data is sparse)."""
    if not segments:
        return 0.5
    for seg in segments:
        if seg.get("word", "").lower() == word.lower():
            return seg.get("confidence", 0.5) or 0.5
    return 0.5


def vote_column(column: dict, model_order: list, segments_by_model: dict) -> str:
    """
    Majority vote for one WTN column. NULL (None) is a valid candidate,
    representing "this word shouldn't be here". Ties are broken by
    average confidence of the tied words' proposing systems, per Fiscus.
    Returns the winning word, or None if NULL wins (word is dropped).
    """
    votes = {}   # word_or_None -> list of models that proposed it
    for model in model_order:
        word = column.get(model)
        votes.setdefault(word, []).append(model)

    # majority = most votes
    max_votes = max(len(models) for models in votes.values())
    tied = [word for word, models in votes.items() if len(models) == max_votes]

    if len(tied) == 1:
        return tied[0]

    # tiebreak: average confidence of the proposing systems for each tied candidate
    best_word, best_conf = None, -1.0
    for word in tied:
        models = votes[word]
        if word is None:
            # NULL's "confidence" - treat as the average of NOT proposing,
            # use a neutral 0.5 baseline so NULL isn't unfairly favoured/disfavoured
            conf = 0.5
        else:
            confs = [
                get_word_confidence(m, word, segments_by_model.get(m, []))
                for m in models
            ]
            conf = sum(confs) / len(confs) if confs else 0.5
        if conf > best_conf:
            best_word, best_conf = word, conf

    return best_word


def rover_combine(hyps: dict, model_order: list, segments_by_model: dict) -> str:
    """Full ROVER pipeline for one sample: build WTN, vote per column,
    concatenate winning words."""
    wtn = build_wtn(hyps, model_order)
    winners = [vote_column(col, model_order, segments_by_model) for col in wtn]
    return " ".join(w for w in winners if w)


# ── Dataset runner ──────────────────────────────────────────────────────────────

def get_indexed_samples(model: str, dataset: str) -> dict:
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def run_dataset(dataset, max_samples=None, rerun=False, split="dev"):
    print(f"\n── {dataset} | ROVER (Fiscus 1997) - PHASE 1: alignment + voting only split={split} ──")

    model_samples = {m: get_indexed_samples(m, dataset) for m in ASR_MODELS}

    indices = get_indices_for_split(dataset, split)
    print(f"  Split: {split} ({len(indices)} samples)")
    if max_samples:
        indices = indices[:max_samples]

    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OLD_OUTPUT_DIR, exist_ok=True)

    filename = f"rover_{dataset}_{split}.json"
    new_output_path = os.path.join(NEW_OUTPUT_DIR, filename)
    old_output_path = os.path.join(OLD_OUTPUT_DIR, filename)

    if os.path.exists(old_output_path) and not rerun:
        with open(old_output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"  Resuming from sample {start_from}/{len(indices)}")
    else:
        results    = []
        start_from = 0

    def save_progress():
        payload = {"progress": len(results), "samples": results}
        with open(new_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        with open(old_output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    for pos in range(start_from, len(indices)):
        idx = indices[pos]
        samples_by_model = {m: model_samples[m].get(idx) for m in ASR_MODELS}

        if not all(samples_by_model.values()):
            results.append({"ref": None, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": "missing sample from one or more models",
                            "dataset_index": idx})
            continue

        ref = samples_by_model["qwen"]["ref"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "ignore_time_segment", "dataset_index": idx})
            continue

        if is_tag_only(ref):
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "skipped": True,
                            "skip_reason": "tag_only_reference", "dataset_index": idx})
            continue

        hyps = {m: samples_by_model[m]["hyp"] for m in ASR_MODELS}
        segments_by_model = {m: samples_by_model[m].get("segments") or [] for m in ASR_MODELS}

        try:
            combined_hyp = rover_combine(hyps, ASR_MODELS, segments_by_model)
        except Exception as e:
            print(f"  ERROR sample {idx}: {e}")
            results.append({"ref": ref, "hyp": None, "severity": None,
                            "sample_WER": None, "error": True,
                            "error_reason": str(e), "dataset_index": idx})
            continue

        sample_wer_val = wer(normalise(ref), normalise(combined_hyp))

        results.append({
            "ref":            ref,
            "hyp":            combined_hyp,
            "source_hyps":    hyps,
            "sample_WER":     sample_wer_val,
            "severity":       None,   # filled in by phase 2
            "dataset_index":  idx,
        })

        if (pos + 1) % 10 == 0:
            save_progress()
            print(f"  {pos+1}/{len(indices)} done")

    valid = [r for r in results if not r.get("skipped") and not r.get("error")
             and r.get("sample_WER") is not None]
    corpus_wer = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None

    output = {
        "approach":       "rover",
        "reference":      "Fiscus 1997 - simple alignment + majority vote, no LLM",
        "asr_models":      ASR_MODELS,
        "phase":          "alignment_and_voting_only - severity not yet judged",
        "dataset":        dataset,
        "split":          split,
        "subset_indices": indices,
        "corpus_wer":     corpus_wer,
        "num_samples":    len(valid),
        "samples":        results,
    }

    with open(new_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open(old_output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "—"
    print(f"\n  WER: {wer_str}  (N={len(valid)})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {old_output_path}")
    print(f"\n  PHASE 1 done. Now run PHASE 2:")
    print(f"  python rerunning/add_severity_to_existing.py --files {new_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     default="commonvoice", choices=DATASETS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split",       default="dev", choices=["dev", "test", "full"])
    parser.add_argument("--rerun",       action="store_true")
    args = parser.parse_args()

    run_dataset(args.dataset, max_samples=args.max_samples,
                rerun=args.rerun, split=args.split)


if __name__ == "__main__":
    main()