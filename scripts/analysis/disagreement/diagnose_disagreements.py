"""
diagnose_disagreements.py

Shows word-level disagreements between Qwen (anchor) and other models
for a sample of clips. Helps understand what the selector will work with.

Usage:
    python scripts/diagnose_disagreements.py --dataset commonvoice --n 5
    python scripts/diagnose_disagreements.py --dataset edacc --n 10
"""

import json
import os
import argparse
from jiwer import process_words

BENCHMARKS_DIR = "benchmarks"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "qwen_english_dialects_20260525_000627.json",
    ("whisper",  "commonvoice"):      "whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_20260525_110315.json",
    ("parakeet", "commonvoice"):      "parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_20260524_234807.json",
    ("wav2vec2", "commonvoice"):      "wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2", "edacc"):            "wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2", "english_dialects"): "wav2vec2_english_dialects_20260526_073439.json",
}

OTHER_MODELS_ALL = ["whisper", "parakeet", "wav2vec2"]

def normalise_word(w: str) -> str:
    """Normalise a word for comparison — lowercase, strip punctuation."""
    import re
    if w in ("[DEL]", None):
        return w
    return re.sub(r"[^\w']", "", w.lower()).strip()

def align_to_qwen(qwen_hyp: str, other_hyp: str) -> dict:
    """
    Align other_hyp to qwen_hyp word by word.
    Returns dict: qwen_word_index -> what other model produced at that position.
    None means deletion (other model skipped this word).
    """
    try:
        out = process_words(qwen_hyp, other_hyp)
        qwen_words = out.references[0]
        other_words = out.hypotheses[0]
        alignments  = out.alignments[0]

        result = {i: None for i in range(len(qwen_words))}

        for op in alignments:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = other_words[hyp_idx] if hyp_idx < len(other_words) else None
            elif op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = other_words[hyp_idx] if hyp_idx < len(other_words) else None
            elif op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    result[i] = "[DEL]"

        return qwen_words, result

    except Exception as e:
        return [], {}

def find_disagreements(qwen_hyp: str, other_hyps: dict, min_disagreements: int = 1) -> list:
    """
    Find positions where at least min_disagreements other models differ from Qwen
    after normalisation. Also notes whether disagreeing models agree on an alternative.
    """
    # align each model to qwen
    alignments = {}
    for model, hyp in other_hyps.items():
        words, alignment = align_to_qwen(qwen_hyp, hyp)
        alignments[model] = (words, alignment)

    if not alignments:
        return []

    first_model = list(alignments.keys())[0]
    qwen_words_aligned, _ = alignments[first_model]

    disagreements = []

    for i, qwen_word in enumerate(qwen_words_aligned):
        qwen_norm = normalise_word(qwen_word)
        alternatives = {}
        n_disagree = 0
        alt_words = []

        for model, (_, alignment) in alignments.items():
            other_word = alignment.get(i, "[DEL]")
            if other_word is None:
                other_word = "[DEL]"

            other_norm = normalise_word(other_word)

            if other_norm != qwen_norm:
                alternatives[model] = other_word
                alt_words.append(other_norm)
                n_disagree += 1
            else:
                alternatives[model] = "✓"

        if n_disagree >= min_disagreements:
            # check if disagreeing models agree on a consensus alternative
            from collections import Counter
            alt_counts = Counter(alt_words)
            top_alt, top_count = alt_counts.most_common(1)[0]
            consensus = top_alt if top_count >= min_disagreements else None

            disagreements.append({
                "position":    i,
                "qwen_word":   qwen_word,
                "alternatives": alternatives,
                "n_disagree":  n_disagree,
                "consensus_alt": consensus,
                "consensus_count": top_count,
            })

    return disagreements

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="edacc",
                        choices=["commonvoice", "edacc", "english_dialects"])
    parser.add_argument("--n",       type=int, default=5,
                        help="Number of samples to show")
    parser.add_argument("--min-disagree", type=int, default=2,
                        help="Minimum number of models that must disagree with Qwen (default: 2)")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Skip clips with no disagreements")
    args = parser.parse_args()

    # load all model files
    model_samples = {}
    for model in ["qwen"] + OTHER_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, args.dataset)])
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    n_total = len(model_samples["qwen"])
    shown   = 0

    for i in range(n_total):
        if shown >= args.n:
            break

        ref      = model_samples["qwen"][i]["ref"]
        qwen_hyp = model_samples["qwen"][i]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        other_hyps = {m: model_samples[m][i]["hyp"] for m in OTHER_MODELS}

        disagreements = find_disagreements(qwen_hyp, other_hyps, args.min_disagree)

        if args.skip_empty and not disagreements:
            continue

        print(f"\n{'='*80}")
        print(f"Clip {i} | {args.dataset}")
        print(f"REF:  {ref[:120]}{'...' if len(ref) > 120 else ''}")
        print(f"QWEN: {qwen_hyp[:120]}{'...' if len(qwen_hyp) > 120 else ''}")
        print(f"\nDisagreements (≥{args.min_disagree} of Whisper/Parakeet differ from Qwen): {len(disagreements)}")

        if disagreements:
            print(f"\n{'Pos':<5} {'Qwen word':<20} {'Whisper':<20} {'Parakeet':<20} {'Consensus alt':<20}")
            print("-" * 90)
            for d in disagreements:
                whisper  = d["alternatives"].get("whisper",  "—")
                parakeet = d["alternatives"].get("parakeet", "—")
                consensus = d["consensus_alt"] if d["consensus_alt"] else "—"
                consensus_str = f"{consensus} ({d['consensus_count']}/2)" if d["consensus_alt"] else "—"
                print(f"{d['position']:<5} {d['qwen_word']:<20} {whisper:<20} {parakeet:<20} {consensus_str:<20}")
        else:
            print("  No disagreements at this threshold.")

        shown += 1

    print(f"\n{'='*70}")
    print(f"Showed {shown} clips from {args.dataset}")

if __name__ == "__main__":
    main()