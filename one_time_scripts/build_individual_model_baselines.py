"""
build_individual_model_baselines.py

Individual ASR model baseline table (mean severity + WER per model, per
dataset, per split) - filters each model's already-scored "merged" full-
pool file (writeup_results/benchmarks/main/{model}_{dataset}_*merged*.json)
down to the CORRECT dev/test indices via get_indices_for_split(), the
same fixed split logic used everywhere else tonight.

Reuses find_canonical_file/load_samples (src.selector) - the same
functions rover.py/mbr_consensus.py already use to pull each model's
own indexed samples, so this stays consistent with the rest of the
project's conventions rather than hand-rolling separate file-finding
logic.

Usage:
    python build_individual_model_baselines.py
"""

import sys
sys.path.insert(0, ".")
from jiwer import wer
from src.judge import normalise
from src.selector import find_canonical_file, load_samples
from src.splits import get_indices_for_split

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "edacc", "english_dialects"]
SPLITS = ["dev", "test"]


def get_indexed_samples(model, dataset):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def main():
    print(f"{'Model':<10}{'Dataset':<20}{'Split':<6}{'N':>6}{'Mean Severity':>16}{'Corpus WER':>12}")
    print("-" * 70)

    for model in ASR_MODELS:
        for dataset in DATASETS:
            indexed = get_indexed_samples(model, dataset)

            for split in SPLITS:
                indices = get_indices_for_split(dataset, split)
                subset = [indexed[i] for i in indices if i in indexed]

                missing = len(indices) - len(subset)
                if missing:
                    print(f"  WARNING: {model}/{dataset}/{split} missing {missing} of {len(indices)} indices in canonical file")

                # same filtering convention as everywhere else tonight -
                # only samples with a real severity/ref/hyp count
                valid = [s for s in subset if s.get("severity") is not None
                        and s.get("ref") and s.get("hyp")]

                n_scored = len(valid)
                mean_sev = sum(s["severity"] for s in valid) / n_scored if n_scored else None

                wer_valid = [s for s in subset if s.get("ref") and s.get("hyp")]
                corpus_wer = wer(
                    [normalise(s["ref"]) for s in wer_valid],
                    [normalise(s["hyp"]) for s in wer_valid]
                ) if wer_valid else None

                sev_str = f"{mean_sev:.3f}" if mean_sev is not None else "-"
                wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
                print(f"{model:<10}{dataset:<20}{split:<6}{n_scored:>6}{sev_str:>16}{wer_str:>12}")


if __name__ == "__main__":
    main()
