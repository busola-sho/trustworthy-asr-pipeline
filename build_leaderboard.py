"""
build_leaderboard.py

Prints only the baseline values needed for the paper's strategy-comparison
table: Qwen3-ASR, WhisperX, Parakeet, Wav2Vec2.0, ROVER, and MBR Consensus.

Individual-model benchmark files are restricted to the requested dev/test
indices before severity and WER are recomputed. ROVER and MBR files are
filtered directly by their stored split. Final columns are macro-averages,
so every dataset receives equal weight.

Usage:
    python build_leaderboard.py --split dev
    python build_leaderboard.py --split test
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jiwer import wer as compute_wer

from src.splits import get_indices_for_split
from src.text_normalise import normalise


DEFAULT_ROOTS = [
    "writeup_results/voting",
    "writeup_results/benchmarks/main",
]

DATASETS = ["commonvoice", "edacc", "english_dialects"]
SYSTEM_ORDER = [
    "Qwen3-ASR",
    "WhisperX",
    "Parakeet",
    "Wav2Vec2.0",
    "ROVER",
    "MBR Consensus",
]

DATASET_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}


def normalize_dataset(raw_name):
    return DATASET_ALIASES.get(raw_name, raw_name)


def normalize_model(raw_name):
    lowered = raw_name.lower()
    if "qwen3-asr" in lowered or "qwen3asr" in lowered:
        return "Qwen3-ASR"
    if "whisperx" in lowered:
        return "WhisperX"
    if "parakeet" in lowered:
        return "Parakeet"
    if "wav2vec2" in lowered:
        return "Wav2Vec2.0"
    return None


def normalize_approach(raw_name):
    lowered = raw_name.lower().replace("-", "_").replace(" ", "_")
    if lowered == "rover":
        return "ROVER"
    if lowered in {"mbr", "mbr_consensus", "mbr_style_consensus"}:
        return "MBR Consensus"
    return None


def load_json_files(roots):
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"WARNING: result root does not exist: {root}")
            continue

        for path in sorted(root_path.rglob("*.json")):
            try:
                with path.open(encoding="utf-8") as file:
                    data = json.load(file)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue

            if isinstance(data, dict):
                yield path, data


def split_baseline_samples(data, dataset, split, path):
    samples = data.get("samples")
    if not samples:
        raise ValueError(
            f"Baseline file has no per-sample data and cannot be restricted "
            f"to split='{split}': {path}"
        )

    split_indices = set(get_indices_for_split(dataset, split))
    subset = []
    for position, sample in enumerate(samples):
        sample_index = sample.get("sample_index")
        if sample_index is None:
            sample_index = position
        if sample_index in split_indices:
            subset.append(sample)
    return subset


def calculate_metrics(samples):
    refs, hyps, severities = [], [], []
    for sample in samples:
        if sample.get("skipped") or sample.get("error"):
            continue

        reference = sample.get("ref")
        hypothesis = sample.get("hyp")
        if reference and hypothesis:
            refs.append(normalise(reference))
            hyps.append(normalise(hypothesis))

        severity = sample.get("severity")
        if severity is not None:
            severities.append(severity)

    corpus_wer = compute_wer(refs, hyps) if refs else None
    mean_severity = sum(severities) / len(severities) if severities else None
    return mean_severity, corpus_wer


def collect_results(roots, split):
    """Return {(system, dataset): result} for the requested split."""
    candidates = defaultdict(list)

    for path, data in load_json_files(roots):
        dataset = normalize_dataset(data.get("dataset"))
        if dataset not in DATASETS:
            continue

        if "approach" not in data and "model" in data:
            system = normalize_model(data.get("model", ""))
            if system is None:
                continue
            samples = split_baseline_samples(data, dataset, split, path)
            mean_severity, corpus_wer = calculate_metrics(samples)

        elif "approach" in data:
            system = normalize_approach(data.get("approach", ""))
            if system is None or data.get("split") != split:
                continue

            # Keep only the standard ROVER and MBR configurations.
            if data.get("percentile") is not None:
                continue

            mean_severity = data.get("mean_severity")
            corpus_wer = data.get("corpus_wer")
        else:
            continue

        candidates[(system, dataset)].append(
            {
                "mean_severity": mean_severity,
                "corpus_wer": corpus_wer,
                "path": str(path),
            }
        )

    results = {}
    for key, matches in candidates.items():
        if len(matches) > 1:
            paths = "\n".join(f"  - {match['path']}" for match in matches)
            system, dataset = key
            raise RuntimeError(
                f"Multiple files found for {system}/{dataset}/{split}. "
                f"Refusing to choose silently:\n{paths}"
            )
        results[key] = matches[0]
    return results


def macro_average(values):
    if len(values) != len(DATASETS) or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def format_severity(value):
    return f"{value:.3f}" if value is not None else "-"


def format_wer(value):
    return f"{value * 100:.2f}%" if value is not None else "-"


def print_results(results, split):
    headers = [
        "System",
        "CommonVoice Sev/WER",
        "EdAcc Sev/WER",
        "English Dialects Sev/WER",
        "Avg Sev",
        "Avg WER",
    ]
    rows = []

    for system in SYSTEM_ORDER:
        severities, wers, dataset_cells = [], [], []
        for dataset in DATASETS:
            result = results.get((system, dataset))
            severity = result.get("mean_severity") if result else None
            wer = result.get("corpus_wer") if result else None
            severities.append(severity)
            wers.append(wer)
            dataset_cells.append(
                f"{format_severity(severity)} / {format_wer(wer)}"
            )

        rows.append(
            [
                system,
                *dataset_cells,
                format_severity(macro_average(severities)),
                format_wer(macro_average(wers)),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row):
        return "  ".join(
            f"{cell:<{widths[index]}}" if index == 0
            else f"{cell:>{widths[index]}}"
            for index, cell in enumerate(row)
        )

    print(f"\nSTRATEGY-COMPARISON BASELINES - {split.upper()} SPLIT")
    print(render(headers))
    print("-" * len(render(headers)))
    for row in rows:
        print(render(row))

    missing = [
        f"{system}/{dataset}"
        for system in SYSTEM_ORDER
        for dataset in DATASETS
        if (system, dataset) not in results
    ]
    if missing:
        print("\nWARNING: missing result cells:")
        for item in missing:
            print(f"  - {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="Split to report (default: dev).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help=(
            "Result roots to scan (default: writeup_results/voting and "
            "writeup_results/benchmarks/main)."
        ),
    )
    args = parser.parse_args()

    results = collect_results(args.roots, args.split)
    print_results(results, args.split)


if __name__ == "__main__":
    main()
