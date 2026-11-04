"""Compare saved ASR baselines on the requested paper split.

Use only the specified sample indices. Empty hypotheses count toward WER and
receive severity 4 when no score exists. Skip references containing only tags.
Average the three dataset results with equal weight."""

from paper_results.reporting.paths import PaperReportFiles

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jiwer import wer as compute_wer

from paper_results.reporting.common import (
    load_split_assignments,
    is_tag_only,
    get_indices_for_split,
    normalise,
)


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


def split_baseline_samples(data, dataset, split, path, paths=None):
    paths = paths or PaperReportFiles()
    samples = data.get("samples")
    if not samples:
        raise ValueError(
            f"Baseline file has no per-sample data and cannot be restricted "
            f"to split={split!r}: {path}"
        )

    split_indices = set(get_indices_for_split(dataset, split, paths=paths))
    subset = []

    for position, sample in enumerate(samples):
        sample_index = sample.get("sample_index")
        if sample_index is None:
            sample_index = sample.get("dataset_index")
        if sample_index is None:
            sample_index = position

        if sample_index in split_indices:
            subset.append(sample)

    return subset


def calculate_metrics(samples):
    references, hypotheses, severities = [], [], []

    for sample in samples:
        if sample.get("skipped") or sample.get("error"):
            continue

        reference = (sample.get("ref") or "").strip()
        if not reference or is_tag_only(reference):
            continue

        hypothesis = (sample.get("hyp") or "").strip()

        # Empty hypotheses are genuine outputs and must remain in WER.
        references.append(normalise(reference))
        hypotheses.append(normalise(hypothesis))

        severity = sample.get("severity")
        # Apply the project's global complete-transcription-failure rule.
        if severity is None and not hypothesis:
            severity = 4
        if severity is not None:
            severities.append(severity)

    corpus_wer = compute_wer(references, hypotheses) if references else None
    mean_severity = sum(severities) / len(severities) if severities else None
    mar = (
        sum(severity >= 2 for severity in severities) / len(severities)
        if severities
        else None
    )
    return mean_severity, corpus_wer, mar


def is_saved_baseline_file(system, path):
    return system in {"ROVER", "MBR Consensus"} and path.parent.name == "baselines"


def collect_results(roots, split, paths=None):
    """Return {(system, dataset): result} for the requested split."""
    paths = paths or PaperReportFiles()
    candidates = defaultdict(list)

    for path, data in load_json_files(roots):
        dataset = normalize_dataset(data.get("dataset"))
        if dataset not in DATASETS:
            continue

        if "approach" not in data and "model" in data:
            system = normalize_model(data.get("model", ""))
            if system is None:
                continue
            samples = split_baseline_samples(data, dataset, split, path, paths=paths)
            mean_severity, corpus_wer, mar = calculate_metrics(samples)

        elif "approach" in data:
            system = normalize_approach(data.get("approach", ""))
            if system is None or not is_saved_baseline_file(system, path):
                continue
            if data.get("split") != split:
                continue
            if data.get("percentile") is not None:
                continue

            # Recompute all three metrics from the same sample set. This avoids
            # mixing a stored mean with a differently filtered MAR denominator.
            samples = data.get("samples")
            if not samples:
                raise ValueError(f"Ensemble file has no per-sample data: {path}")
            mean_severity, corpus_wer, mar = calculate_metrics(samples)

        else:
            continue

        candidates[(system, dataset)].append(
            {
                "mean_severity": mean_severity,
                "corpus_wer": corpus_wer,
                "mar": mar,
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


def format_percentage(value):
    return f"{value * 100:.2f}%" if value is not None else "-"


def print_results(results, split):
    headers = [
        "System",
        "CommonVoice Sev/WER",
        "EdAcc Sev/WER",
        "English Dialects Sev/WER",
        "Avg Sev",
        "Avg WER",
        "Avg MAR",
    ]
    rows = []

    for system in SYSTEM_ORDER:
        severities, wers, mars, dataset_cells = [], [], [], []
        for dataset in DATASETS:
            result = results.get((system, dataset))
            severity = result.get("mean_severity") if result else None
            wer = result.get("corpus_wer") if result else None
            mar = result.get("mar") if result else None
            severities.append(severity)
            wers.append(wer)
            mars.append(mar)
            dataset_cells.append(
                f"{format_severity(severity)} / {format_percentage(wer)}"
            )

        rows.append(
            [
                system,
                *dataset_cells,
                format_severity(macro_average(severities)),
                format_percentage(macro_average(wers)),
                format_percentage(macro_average(mars)),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row):
        return "  ".join(
            f"{cell:<{widths[index]}}" if index == 0 else f"{cell:>{widths[index]}}"
            for index, cell in enumerate(row)
        )

    print(f"\nSTRATEGY-COMPARISON BASELINES - {split.upper()} SPLIT")
    print("MAR = percentage of scored samples with severity >= 2")
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


def main(paths=None):
    paths = paths or PaperReportFiles()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="Split to report (default: dev).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help=("Explicit result roots to scan; use the curated baselines directory."),
    )
    parser.add_argument("--splits-file", type=Path, required=True)
    args = parser.parse_args()
    paths = PaperReportFiles(split_assignments=load_split_assignments(args.splits_file))
    results = collect_results(args.roots, args.split, paths=paths)
    print_results(results, args.split)


if __name__ == "__main__":
    raise SystemExit("Use python -m asr_pipeline report with a separate --output-dir")
