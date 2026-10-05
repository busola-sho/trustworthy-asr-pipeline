"""Extract paper-ready Police Scotland results from the portable pipeline.

The individual-model file contains severity judgements but no WER values.
The fusion evaluation file contains per-sample WER, so the reported WER is
the mean of those sample-level WER values, not corpus WER.

Usage:
    python extract_police_scotland_results.py \
        --individual individual_model_severity.json \
        --fusion eval_results_public.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


MODEL_DISPLAY = {
    "qwen": "Qwen3-ASR",
    "whisperx": "WhisperX",
    "parakeet": "Parakeet",
    "wav2vec2": "Wav2Vec2.0",
}
MODEL_ORDER = ["qwen", "whisperx", "parakeet", "wav2vec2"]


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def severity_metrics(values):
    values = [int(value) for value in values if value is not None]
    if not values:
        return None

    n = len(values)
    meaning_altering = sum(value >= 2 for value in values)
    return {
        "n": n,
        "mean_severity": sum(values) / n,
        "mar": meaning_altering / n,
        "preservation": (n - meaning_altering) / n,
        "distribution": Counter(values),
    }


def individual_results(data):
    rows = data.get("rows", [])
    grouped = defaultdict(list)

    for row in rows:
        model = row.get("model")
        if model:
            grouped[model].append(row.get("severity"))

    results = {}
    for model in MODEL_ORDER:
        metrics = severity_metrics(grouped.get(model, []))
        if metrics is not None:
            results[model] = metrics

    return results


def fusion_result(data):
    valid_samples = [
        sample
        for sample in data.get("samples", [])
        if sample.get("error") is None and sample.get("severity") is not None
    ]

    metrics = severity_metrics(
        [sample.get("severity") for sample in valid_samples]
    )
    if metrics is None:
        return None

    wers = [
        float(sample["wer"])
        for sample in valid_samples
        if sample.get("wer") is not None
    ]
    metrics["mean_sample_wer"] = sum(wers) / len(wers) if wers else None
    metrics["wer_n"] = len(wers)
    metrics["pipeline"] = data.get("pipeline")
    metrics["judge"] = data.get("judge")
    return metrics


def format_percent(value):
    return f"{value * 100:.1f}%" if value is not None else "-"


def print_row(label, metrics):
    print(
        f"{label:<20} "
        f"severity={metrics['mean_severity']:.3f}, "
        f"MAR={format_percent(metrics['mar'])}, "
        f"preserved={format_percent(metrics['preservation'])}, "
        f"N={metrics['n']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--individual",
        required=True,
        help="Path to individual_model_severity JSON.",
    )
    parser.add_argument(
        "--fusion",
        required=True,
        help="Path to eval_results_public JSON.",
    )
    args = parser.parse_args()

    individual_data = load_json(args.individual)
    fusion_data = load_json(args.fusion)

    individual = individual_results(individual_data)
    fusion = fusion_result(fusion_data)

    print("POLICE SCOTLAND SEVERITY RESULTS")
    print("MAR = percentage with severity >= 2")
    print("Preserved = percentage with severity 0-1\n")

    for model in MODEL_ORDER:
        if model in individual:
            print_row(MODEL_DISPLAY[model], individual[model])
        else:
            print(f"{MODEL_DISPLAY[model]:<20} NOT FOUND")

    if fusion is None:
        print(f"{'Unanchored Fusion':<20} NOT FOUND")
        return

    print_row("Unanchored Fusion", fusion)

    print("\nWER AVAILABILITY")
    print("Individual systems: unavailable in individual_model_severity JSON")
    if fusion["mean_sample_wer"] is None:
        print("Unanchored Fusion: unavailable")
    else:
        print(
            "Unanchored Fusion: "
            f"mean sample WER={fusion['mean_sample_wer'] * 100:.2f}% "
            f"(N={fusion['wer_n']})"
        )
        print("Note: this is mean sample WER, not corpus WER.")

    best_mar_model = min(individual, key=lambda model: individual[model]["mar"])
    best_severity_model = min(
        individual,
        key=lambda model: individual[model]["mean_severity"],
    )

    print("\nBEST INDIVIDUAL COMPARISONS")
    print(
        "Lowest individual MAR: "
        f"{MODEL_DISPLAY[best_mar_model]} "
        f"({format_percent(individual[best_mar_model]['mar'])})"
    )
    print(
        "Lowest individual mean severity: "
        f"{MODEL_DISPLAY[best_severity_model]} "
        f"({individual[best_severity_model]['mean_severity']:.3f})"
    )
    print(
        "Meaning preservation gain over best individual MAR: "
        f"{format_percent(fusion['preservation'] - individual[best_mar_model]['preservation'])}"
    )


if __name__ == "__main__":
    main()

