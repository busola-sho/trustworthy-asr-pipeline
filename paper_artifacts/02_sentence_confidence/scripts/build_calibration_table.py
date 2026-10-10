import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = PACKAGE_ROOT / "calibration"

DISPLAY = {
    "commonvoice_test": "Common Voice",
    "edacc_test": "EdAcc",
    "english_dialects_test": "English Dialects",
    "shetland_full": "Shetland",
    "police_scotland": "Police Scotland",
}


def main():
    source = CALIBRATION_DIR / "calibration_results.csv"

    with open(source, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []

    for row in rows:
        output_rows.append({
            "Dataset": DISPLAY[row["dataset"]],
            "Raw ECE": round(float(row["raw_ece"]), 3),
            "Calibrated ECE": round(
                float(row["calibrated_ece"]),
                3,
            ),
            "Raw Brier": round(float(row["raw_brier"]), 3),
            "Calibrated Brier": round(
                float(row["calibrated_brier"]),
                3,
            ),
        })

    csv_path = CALIBRATION_DIR / "calibration_paper_table.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    txt_path = CALIBRATION_DIR / "calibration_paper_table.txt"

    header = (
        f"{'Dataset':<20}"
        f"{'Raw ECE':>10}"
        f"{'Cal. ECE':>11}"
        f"{'Raw Brier':>12}"
        f"{'Cal. Brier':>13}"
    )

    lines = [
        "VERBALISED CONFSCORE CALIBRATION",
        "",
        header,
        "-" * len(header),
    ]

    for row in output_rows:
        lines.append(
            f"{row['Dataset']:<20}"
            f"{row['Raw ECE']:>10.3f}"
            f"{row['Calibrated ECE']:>11.3f}"
            f"{row['Raw Brier']:>12.3f}"
            f"{row['Calibrated Brier']:>13.3f}"
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
