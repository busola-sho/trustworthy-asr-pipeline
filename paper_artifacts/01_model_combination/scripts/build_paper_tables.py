import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
BASELINE_DIR = ROOT / "data" / "baselines"
GRID_DIR = ROOT / "data" / "grid"
TABLE_DIR = ROOT / "tables"

DATASETS = [
    "commonvoice",
    "edacc",
    "english_dialects",
]

BASELINE_SYSTEMS = [
    "Qwen3-ASR",
    "WhisperX",
    "Parakeet",
    "Wav2Vec2.0",
    "ROVER",
    "MBR Consensus",
]

STRATEGIES = {
    "Selection": "selection_naive",
    "Unanchored Fusion": "unanchored_fusion_naive",
    "Anchored Correction": "anchored_correction_naive",
}

GUIDANCE_CONDITIONS = {
    "None": "unanchored_fusion_naive",
    "Manual": "unanchored_fusion_context_v1",
    "Auto": "unanchored_fusion_context_v2",
}

sys.path.insert(0, str(SCRIPTS_DIR))

from build_leaderboard import collect_results


def macro_average(values):
    if len(values) != len(DATASETS):
        raise RuntimeError(f"Incomplete values: {values}")

    if any(value is None for value in values):
        raise RuntimeError(f"Missing value: {values}")

    return sum(values) / len(values)


def find_mar(result):
    for key, value in result.items():
        lowered = key.lower()

        if (
            key == "mar"
            or "alteration" in lowered
            or "flag_rate" in lowered
        ):
            if value is None:
                return None

            value = float(value)

            return value / 100 if value > 1 else value

    raise RuntimeError(
        f"Could not locate MAR in result keys: {result.keys()}"
    )


def calculate_mar(samples):
    severities = [
        sample["severity"]
        for sample in samples
        if sample.get("severity") is not None
        and not sample.get("skipped")
        and not sample.get("error")
    ]

    if not severities:
        return None

    return sum(
        severity >= 2
        for severity in severities
    ) / len(severities)


def load_grid_result(directory, dataset, split):
    path = (
        GRID_DIR
        / directory
        / f"{directory}_{dataset}_gemma4_{split}.json"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    mean_severity = data.get("mean_severity")
    corpus_wer = data.get("corpus_wer")
    mar = calculate_mar(data.get("samples", []))

    if mean_severity is None:
        raise RuntimeError(f"Missing mean_severity in {path}")

    if corpus_wer is None:
        raise RuntimeError(f"Missing corpus_wer in {path}")

    if mar is None:
        raise RuntimeError(f"Could not calculate MAR from {path}")

    return {
        "mean_severity": float(mean_severity),
        "corpus_wer": float(corpus_wer),
        "mar": float(mar),
    }


def macro_grid_result(directory, split):
    results = [
        load_grid_result(
            directory,
            dataset,
            split,
        )
        for dataset in DATASETS
    ]

    return {
        "severity": macro_average(
            [result["mean_severity"] for result in results]
        ),
        "wer": macro_average(
            [result["corpus_wer"] for result in results]
        ),
        "mar": macro_average(
            [result["mar"] for result in results]
        ),
    }


def macro_baseline_result(results, system):
    system_results = []

    for dataset in DATASETS:
        key = (system, dataset)

        if key not in results:
            raise RuntimeError(
                f"Missing baseline result: {system}/{dataset}"
            )

        result = results[key]

        system_results.append(
            {
                "severity": result["mean_severity"],
                "wer": result["corpus_wer"],
                "mar": find_mar(result),
            }
        )

    return {
        "severity": macro_average(
            [result["severity"] for result in system_results]
        ),
        "wer": macro_average(
            [result["wer"] for result in system_results]
        ),
        "mar": macro_average(
            [result["mar"] for result in system_results]
        ),
    }


def build_strategy_rows():
    split_results = {
        split: collect_results(
            [str(BASELINE_DIR)],
            split,
        )
        for split in ["dev", "test"]
    }

    rows = []

    for system in BASELINE_SYSTEMS:
        dev = macro_baseline_result(
            split_results["dev"],
            system,
        )
        test = macro_baseline_result(
            split_results["test"],
            system,
        )

        rows.append(
            {
                "system": system,
                "dev_severity": dev["severity"],
                "dev_wer_pct": dev["wer"] * 100,
                "dev_mar_pct": dev["mar"] * 100,
                "test_severity": test["severity"],
                "test_wer_pct": test["wer"] * 100,
                "test_mar_pct": test["mar"] * 100,
            }
        )

    for system, directory in STRATEGIES.items():
        dev = macro_grid_result(directory, "dev")
        test = macro_grid_result(directory, "test")

        rows.append(
            {
                "system": system,
                "dev_severity": dev["severity"],
                "dev_wer_pct": dev["wer"] * 100,
                "dev_mar_pct": dev["mar"] * 100,
                "test_severity": test["severity"],
                "test_wer_pct": test["wer"] * 100,
                "test_mar_pct": test["mar"] * 100,
            }
        )

    return rows


def build_guidance_rows():
    rows = []

    for condition, directory in GUIDANCE_CONDITIONS.items():
        dev = macro_grid_result(directory, "dev")
        test = macro_grid_result(directory, "test")

        rows.append(
            {
                "condition": condition,
                "dev_severity": dev["severity"],
                "dev_wer_pct": dev["wer"] * 100,
                "dev_mar_pct": dev["mar"] * 100,
                "test_severity": test["severity"],
                "test_wer_pct": test["wer"] * 100,
                "test_mar_pct": test["mar"] * 100,
            }
        )

    return rows


def write_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, rows, name_key):
    header = (
        f"{'Condition/System':<24}"
        f"{'Dev Sev':>10}"
        f"{'Dev WER':>11}"
        f"{'Dev MAR':>11}"
        f"{'Test Sev':>11}"
        f"{'Test WER':>12}"
        f"{'Test MAR':>12}"
    )

    lines = [
        header,
        "-" * len(header),
    ]

    for row in rows:
        lines.append(
            f"{row[name_key]:<24}"
            f"{row['dev_severity']:>10.3f}"
            f"{row['dev_wer_pct']:>10.2f}%"
            f"{row['dev_mar_pct']:>10.2f}%"
            f"{row['test_severity']:>11.3f}"
            f"{row['test_wer_pct']:>11.2f}%"
            f"{row['test_mar_pct']:>11.2f}%"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    strategy_rows = build_strategy_rows()
    guidance_rows = build_guidance_rows()

    strategy_csv = TABLE_DIR / "strategy_comparison.csv"
    strategy_txt = TABLE_DIR / "strategy_comparison.txt"
    guidance_csv = TABLE_DIR / "prompt_guidance_ablation.csv"
    guidance_txt = TABLE_DIR / "prompt_guidance_ablation.txt"

    write_csv(strategy_csv, strategy_rows)
    write_text(
        strategy_txt,
        strategy_rows,
        "system",
    )

    write_csv(guidance_csv, guidance_rows)
    write_text(
        guidance_txt,
        guidance_rows,
        "condition",
    )

    print(strategy_txt.read_text())
    print(guidance_txt.read_text())

    print("Saved:")
    print(f"  {strategy_csv}")
    print(f"  {strategy_txt}")
    print(f"  {guidance_csv}")
    print(f"  {guidance_txt}")


if __name__ == "__main__":
    main()
