"""
Extract final held-out test and Shetland results for the paper.

Reports mean severity, corpus WER, meaning-alteration rate (MAR), and the
number of samples with severity judgements. Severity >= 2 is meaning-altering.

The three held-out test datasets use the verified final result directories:
  - ROVER: writeup_results/voting/rover
  - MBR: writeup_results/clean_mbr_consensus
  - LLM strategies: writeup_results/clean_grid (naive conditions)

Shetland has no dev/test split and uses the available full-set runs. Selection
was not run on Shetland. The available anchored Shetland run used the earlier
manual-guidance condition, so it is labelled explicitly.

Usage:
    python extract_all_methods_test_shetland.py
"""

import glob
import json
from collections import Counter
from pathlib import Path

from jiwer import wer as compute_wer

from src.splits import get_indices_for_split
from src.text_normalise import normalise


TEST_DATASETS = ["commonvoice", "edacc", "english_dialects"]
DATASET_DISPLAY = {
    "commonvoice": "Common Voice",
    "edacc": "EdAcc",
    "english_dialects": "English Dialects",
    "shetland": "Shetland",
}

# "Best" here means lowest mean severity among the individual systems.
BEST_MODEL_PER_DATASET = {
    "commonvoice": "parakeet",
    "edacc": "qwen",
    "english_dialects": "whisperx",
    "shetland": "qwen",
}

SHETLAND_BASELINES = {
    "qwen": "results/benchmarks/shetland/shetland_qwen3asr_20260603_150124.json",
    "whisperx": "results/benchmarks/shetland/shetland_whisper_20260603_123115.json",
    "parakeet": "results/benchmarks/shetland/shetland_parakeet_20260606_134131.json",
    "wav2vec2": "results/benchmarks/shetland/shetland_wav2vec2_20260606_134507.json",
}


def load_json(path):
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_unscoreable_reference(reference):
    if reference is None:
        return True
    stripped = reference.strip().upper()
    return (
        not stripped
        or "IGNORE_TIME_SEGMENT_IN_SCORING" in stripped
        or (stripped.startswith("<") and stripped.endswith(">"))
    )


def calculate_metrics(samples):
    """Return mean severity, corpus WER, MAR, judged N, and WER N.

    Empty-string hypotheses are retained for WER and therefore count as full
    deletions. Only None hypotheses, explicitly skipped/error samples, and
    unscoreable references are excluded.
    """
    references = []
    hypotheses = []
    severities = []

    for sample in samples:
        if sample.get("skipped") or sample.get("error"):
            continue

        reference = sample.get("ref")
        hypothesis = sample.get("hyp")

        if not is_unscoreable_reference(reference) and hypothesis is not None:
            references.append(normalise(reference))
            hypotheses.append(normalise(hypothesis))

        severity = sample.get("severity")
        if severity is not None:
            severities.append(int(severity))

    mean_severity = (
        sum(severities) / len(severities) if severities else None
    )
    corpus_wer = (
        compute_wer(references, hypotheses) if references else None
    )
    mar = (
        sum(severity >= 2 for severity in severities) / len(severities)
        if severities
        else None
    )

    return {
        "mean_severity": mean_severity,
        "corpus_wer": corpus_wer,
        "mar": mar,
        "severity_n": len(severities),
        "wer_n": len(references),
    }


def metrics_from_file(path, dataset=None, split=None):
    data = load_json(path)
    if data is None:
        return None

    samples = data.get("samples", [])
    if split in {"dev", "test"}:
        split_indices = set(get_indices_for_split(dataset, split))
        restricted = []
        for position, sample in enumerate(samples):
            sample_index = sample.get("sample_index")
            if sample_index is None:
                sample_index = sample.get("dataset_index", position)
            if sample_index in split_indices:
                restricted.append(sample)
        samples = restricted

    return calculate_metrics(samples)


def find_baseline_model_path(model, dataset):
    if dataset == "shetland":
        return SHETLAND_BASELINES.get(model)

    patterns = {
        "qwen": [
            f"writeup_results/benchmarks/main/qwen3asr_{dataset}_*.json",
            f"writeup_results/benchmarks/main/qwen_{dataset}_*.json",
        ],
        "whisperx": [
            f"writeup_results/benchmarks/main/whisperx_{dataset}_*.json",
        ],
        "parakeet": [
            f"writeup_results/benchmarks/main/parakeet_{dataset}_*.json",
        ],
        "wav2vec2": [
            f"writeup_results/benchmarks/main/wav2vec2_{dataset}_*.json",
        ],
    }

    matches = []
    for pattern in patterns[model]:
        matches.extend(glob.glob(pattern))

    matches = sorted(
        path
        for path in set(matches)
        if "sub150" not in path
        and "sub100" not in path
        and "whisper_ft_chunked" not in path
    )

    if not matches:
        return None

    merged = [path for path in matches if "merged" in Path(path).name]
    if len(merged) == 1:
        return merged[0]

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Multiple baseline files found for {model}/{dataset}. "
        "Refusing to choose silently:\n  - " + "\n  - ".join(matches)
    )


def best_individual_metrics(dataset, split):
    model = BEST_MODEL_PER_DATASET[dataset]
    path = find_baseline_model_path(model, dataset)
    metrics = metrics_from_file(path, dataset, split)
    return metrics, path, model


def format_result(metrics):
    if metrics is None:
        return "NOT FOUND"

    severity = metrics["mean_severity"]
    corpus_wer = metrics["corpus_wer"]
    mar = metrics["mar"]

    severity_text = f"{severity:.3f}" if severity is not None else "-"
    wer_text = f"{corpus_wer * 100:.2f}%" if corpus_wer is not None else "-"
    mar_text = f"{mar * 100:.1f}%" if mar is not None else "-"

    return (
        f"severity={severity_text}, WER={wer_text}, MAR={mar_text}, "
        f"N={metrics['severity_n']}"
    )


def run_file_method(name, test_path, shetland_path):
    print(f"\n{name}:")

    for dataset in TEST_DATASETS:
        path = test_path(dataset)
        metrics = metrics_from_file(path)
        print(f"  {DATASET_DISPLAY[dataset]}: {format_result(metrics)}")

    path = shetland_path()
    metrics = metrics_from_file(path)
    print(f"  Shetland: {format_result(metrics)}")


def main():
    print("FINAL TEST + SHETLAND RESULTS")
    print("MAR = percentage of judged samples with severity >= 2")

    print("\nBest individual model (selected independently per dataset by severity):")
    for dataset in TEST_DATASETS:
        metrics, _, model = best_individual_metrics(dataset, "test")
        print(
            f"  {DATASET_DISPLAY[dataset]} [{model}]: "
            f"{format_result(metrics)}"
        )
    metrics, _, model = best_individual_metrics("shetland", "full")
    print(f"  Shetland [{model}]: {format_result(metrics)}")

    run_file_method(
        "ROVER",
        lambda dataset: (
            f"writeup_results/voting/rover/rover_{dataset}_test.json"
        ),
        lambda: "writeup_results/voting/rover/rover_shetland_full.json",
    )

    run_file_method(
        "Pairwise-WER consensus (MBR-style)",
        lambda dataset: (
            f"writeup_results/clean_mbr_consensus/mbr_{dataset}_test.json"
        ),
        lambda: (
            "writeup_results/ensembles/mbr_consensus/"
            "mbr_shetland_full.json"
        ),
    )

    run_file_method(
        "Candidate selection (naive prompt)",
        lambda dataset: (
            "writeup_results/clean_grid/selection_naive/"
            f"selection_naive_{dataset}_gemma4_test.json"
        ),
        lambda: None,
    )

    run_file_method(
        "Anchored correction (naive on test; manual guidance on Shetland)",
        lambda dataset: (
            "writeup_results/clean_grid/anchored_correction_naive/"
            f"anchored_correction_naive_{dataset}_gemma4_test.json"
        ),
        lambda: (
            "writeup_results/ensembles/context_v1/gemma4/"
            "context_shetland_gemma4_full.json"
        ),
    )

    run_file_method(
        "Unanchored fusion (naive prompt)",
        lambda dataset: (
            "writeup_results/clean_grid/unanchored_fusion_naive/"
            f"unanchored_fusion_naive_{dataset}_gemma4_test.json"
        ),
        lambda: (
            "writeup_results/ensembles/naive/gemma4/"
            "naive_shetland_gemma4sel_full.json"
        ),
    )


if __name__ == "__main__":
    main()
