import csv
import importlib.util
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
TABLES_DIR = PACKAGE_ROOT / "tables"

sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

spec = importlib.util.spec_from_file_location(
    "build_leaderboard",
    PACKAGE_ROOT / "scripts" / "build_leaderboard.py",
)
leaderboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(leaderboard)


DATASETS = [
    "commonvoice",
    "edacc",
    "english_dialects",
]

DISPLAY = {
    "commonvoice": "Common Voice",
    "edacc": "EdAcc",
    "english_dialects": "English Dialects",
    "shetland": "Shetland",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def mar_from_data(data):
    samples = [
        sample
        for sample in data.get("samples", [])
        if sample.get("severity") is not None
        and not sample.get("skipped")
        and not sample.get("error")
    ]

    if samples:
        return sum(
            sample["severity"] >= 2
            for sample in samples
        ) / len(samples)

    distribution = data.get("severity_distribution", {})
    counts = {
        int(key): value
        for key, value in distribution.items()
    }
    total = sum(counts.values())

    if not total:
        return None

    return sum(
        count
        for severity, count in counts.items()
        if severity >= 2
    ) / total


def metrics_from_file(path):
    data = load_json(path)

    valid_samples = [
        sample
        for sample in data.get("samples", [])
        if sample.get("severity") is not None
        and not sample.get("skipped")
        and not sample.get("error")
    ]

    severity = data.get("mean_severity")
    if severity is None and valid_samples:
        severity = sum(
            sample["severity"]
            for sample in valid_samples
        ) / len(valid_samples)

    return {
        "severity": severity,
        "wer": data.get("corpus_wer"),
        "mar": mar_from_data(data),
        "n": data.get("num_samples") or len(valid_samples),
    }


def value_from_result(result, *names):
    for name in names:
        if result.get(name) is not None:
            return result[name]
    return None


def raw_qwen_results(split):
    results = leaderboard.collect_results(
        [DATA_ROOT / "baselines"],
        split,
    )

    output = {}

    for dataset in DATASETS:
        result = results[("Qwen3-ASR", dataset)]

        output[dataset] = {
            "severity": value_from_result(
                result,
                "mean_severity",
                "severity",
            ),
            "wer": value_from_result(
                result,
                "corpus_wer",
                "wer",
            ),
            "mar": value_from_result(
                result,
                "mar",
                "mar_rate",
            ),
            "n": value_from_result(
                result,
                "n",
                "num_samples",
            ),
        }

    return output


def find_one(folder, pattern):
    matches = sorted(folder.glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one match for {pattern}, "
            f"found {len(matches)}"
        )

    return matches[0]


def collect_rows():
    rows = []

    postedit_dir = DATA_ROOT / "single_hypothesis_postedit"
    grid_dir = DATA_ROOT / "grid"

    for split in ["dev", "test"]:
        raw_results = raw_qwen_results(split)

        for dataset in DATASETS:
            postedit_path = find_one(
                postedit_dir,
                f"*_{dataset}_gemma4_{split}.json",
            )

            fusion_path = (
                grid_dir
                / "unanchored_fusion_naive"
                / (
                    "unanchored_fusion_naive_"
                    f"{dataset}_gemma4_{split}.json"
                )
            )

            systems = [
                ("Raw Qwen", raw_results[dataset]),
                ("Post-edit", metrics_from_file(postedit_path)),
                (
                    "Unanchored Fusion",
                    metrics_from_file(fusion_path),
                ),
            ]

            for system, metrics in systems:
                rows.append({
                    "Dataset": DISPLAY[dataset],
                    "Split": split,
                    "System": system,
                    "Mean severity": metrics["severity"],
                    "WER": metrics["wer"],
                    "MAR": metrics["mar"],
                    "N": metrics["n"],
                })

    shetland_paths = {
        "Raw Qwen": (
            DATA_ROOT
            / "shetland"
            / "shetland_qwen3asr_20260603_150124.json"
        ),
        "Post-edit": find_one(
            postedit_dir,
            "*_shetland_gemma4_full.json",
        ),
        "Unanchored Fusion": (
            DATA_ROOT
            / "shetland"
            / "naive_shetland_gemma4sel_full.json"
        ),
    }

    for system, result_path in shetland_paths.items():
        metrics = metrics_from_file(result_path)

        rows.append({
            "Dataset": "Shetland",
            "Split": "full",
            "System": system,
            "Mean severity": metrics["severity"],
            "WER": metrics["wer"],
            "MAR": metrics["mar"],
            "N": metrics["n"],
        })

    return rows


def format_number(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def format_percentage(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def main():
    rows = collect_rows()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / "single_hypothesis_postedit.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    txt_path = TABLES_DIR / "single_hypothesis_postedit.txt"

    header = (
        f"{'Dataset':<18}"
        f"{'Split':<7}"
        f"{'System':<22}"
        f"{'Severity':>10}"
        f"{'WER':>10}"
        f"{'MAR':>10}"
    )

    lines = [
        "SINGLE-HYPOTHESIS POST-EDITING COMPARISON",
        "",
        header,
        "-" * len(header),
    ]

    for row in rows:
        lines.append(
            f"{row['Dataset']:<18}"
            f"{row['Split']:<7}"
            f"{row['System']:<22}"
            f"{format_number(row['Mean severity']):>10}"
            f"{format_percentage(row['WER']):>10}"
            f"{format_percentage(row['MAR']):>10}"
        )

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
