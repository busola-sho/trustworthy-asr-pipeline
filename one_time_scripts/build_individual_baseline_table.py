"""
build_individual_baseline_table.py

Cross-dataset macro-averaged individual ASR baseline table - mean
severity, meaning-altered rate (severity >= 2), and corpus WER, per
model per split, averaged across the 3 datasets (macro-average: mean
of per-dataset means, NOT pooled - matching the same convention
established earlier tonight for the 9-cell grid's Alt% column, to
avoid English Dialects' large N dominating the average).

Usage:
    python build_individual_baseline_table.py
"""

import sys
sys.path.insert(0, ".")
from jiwer import wer
from src.judge import normalise
from src.selector import find_canonical_file, load_samples
from src.splits import get_indices_for_split

ASR_MODELS = ["qwen", "whisperx", "parakeet", "wav2vec2"]
MODEL_DISPLAY = {"qwen": "Qwen3-ASR", "whisperx": "WhisperX", "parakeet": "Parakeet", "wav2vec2": "Wav2Vec2.0"}
DATASETS = ["commonvoice", "edacc", "english_dialects"]
SPLITS = ["dev", "test"]


def get_indexed_samples(model, dataset):
    path = find_canonical_file(model, dataset)
    samples = load_samples(path)
    return {s["sample_index"]: s for s in samples if s.get("sample_index") is not None}


def per_dataset_stats(model, dataset, split):
    indexed = get_indexed_samples(model, dataset)
    indices = get_indices_for_split(dataset, split)
    subset = [indexed[i] for i in indices if i in indexed]

    valid = [s for s in subset if s.get("severity") is not None and s.get("ref") and s.get("hyp")]
    n_scored = len(valid)
    mean_sev = sum(s["severity"] for s in valid) / n_scored if n_scored else None
    alt_rate = sum(1 for s in valid if s["severity"] >= 2) / n_scored if n_scored else None

    wer_valid = [s for s in subset if s.get("ref") and s.get("hyp")]
    corpus_wer = wer(
        [normalise(s["ref"]) for s in wer_valid],
        [normalise(s["hyp"]) for s in wer_valid]
    ) if wer_valid else None

    return mean_sev, alt_rate, corpus_wer, n_scored


def main():
    print(f"{'Split':<12}{'Model':<12}{'Mean Sev':>10}{'Altered %':>12}{'Corpus WER %':>14}")
    print("-" * 62)

    for split in SPLITS:
        for model in ASR_MODELS:
            sevs, alts, wers = [], [], []
            for dataset in DATASETS:
                mean_sev, alt_rate, corpus_wer, n = per_dataset_stats(model, dataset, split)
                if mean_sev is not None:
                    sevs.append(mean_sev)
                if alt_rate is not None:
                    alts.append(alt_rate)
                if corpus_wer is not None:
                    wers.append(corpus_wer)

            macro_sev = sum(sevs) / len(sevs) if sevs else None
            macro_alt = sum(alts) / len(alts) * 100 if alts else None
            macro_wer = sum(wers) / len(wers) * 100 if wers else None

            print(f"{split:<12}{MODEL_DISPLAY[model]:<12}{macro_sev:>10.3f}{macro_alt:>12.1f}{macro_wer:>14.2f}")


if __name__ == "__main__":
    main()
