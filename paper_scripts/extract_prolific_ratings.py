"""
Extract and combine the original judge ratings with the Prolific ratings.

Outputs:
1. samples_with_judge_ratings.csv
2. participant_ratings_long.csv
3. judge_and_participant_ratings.csv
4. prolific_ratings_complete.xlsx
"""

import argparse
import re
from pathlib import Path

import pandas as pd


CURRENT_STUDY_ID = "6abe185f7296e8fea1fc642a"


def find_column(headers, name):
    matches = [
        index
        for index, value in enumerate(headers)
        if str(value).strip() == name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one column called {name!r}, found {matches}"
        )

    return matches[0]


def extract_participant_ratings(qualtrics_path):
    raw = pd.read_excel(qualtrics_path, header=None)

    machine_headers = raw.iloc[0]
    question_labels = raw.iloc[1]
    responses = raw.iloc[2:].copy()

    study_col = find_column(machine_headers, "STUDY_ID")
    participant_col = find_column(machine_headers, "PROLIFIC_PID")
    session_col = find_column(machine_headers, "SESSION_ID")
    set_col = find_column(machine_headers, "set")
    response_col = find_column(machine_headers, "ResponseId")
    recorded_col = find_column(machine_headers, "RecordedDate")
    duration_col = find_column(
        machine_headers,
        "Duration (in seconds)",
    )
    attention_col = find_column(machine_headers, "Question")

    responses = responses[
        responses[study_col].astype(str) == CURRENT_STUDY_ID
    ].copy()

    responses["_recorded_date"] = pd.to_datetime(
        responses[recorded_col]
    )
    responses["_set_id"] = pd.to_numeric(
        responses[set_col],
        errors="raise",
    ).astype(int)

    # Set 13 received two submissions. Keeping the latest response also
    # retains the replacement participant rather than the earlier,
    # exceptionally fast submission.
    responses = (
        responses
        .sort_values("_recorded_date")
        .groupby("_set_id", as_index=False)
        .tail(1)
        .sort_values("_set_id")
    )

    if len(responses) != 15:
        raise RuntimeError(
            f"Expected 15 retained participants, found {len(responses)}"
        )

    rating_columns = []

    for column in range(raw.shape[1]):
        label = str(question_labels.iloc[column])
        machine_name = str(machine_headers.iloc[column])

        sample_match = re.match(
            r"^(J\d+)\s+-\s+Compare",
            label,
        )

        if sample_match and "Questions" in machine_name:
            position_match = re.match(
                r"^(\d+)_Questions",
                machine_name,
            )

            if not position_match:
                raise RuntimeError(
                    f"Could not extract presentation order from "
                    f"{machine_name!r}"
                )

            rating_columns.append(
                {
                    "column": column,
                    "study_id": sample_match.group(1),
                    "presentation_order": int(
                        position_match.group(1)
                    ),
                }
            )

    rows = []

    for _, response in responses.iterrows():
        set_id = int(response["_set_id"])
        participant_id = str(response[participant_col])
        response_id = str(response[response_col])
        session_id = str(response[session_col])
        duration = int(response[duration_col])
        attention_check = str(response[attention_col])

        if attention_check != "Meaning only":
            raise RuntimeError(
                f"Participant {participant_id} failed the "
                f"attention check: {attention_check!r}"
            )

        participant_rows = []

        for question in rating_columns:
            column = question["column"]
            answer = response[column]

            if pd.isna(answer):
                continue

            rating_match = re.match(
                r"^\s*([0-4])\b",
                str(answer),
            )

            if not rating_match:
                raise RuntimeError(
                    f"Could not parse rating {answer!r}"
                )

            def timing_value(offset):
                value = response[column + offset]

                if pd.isna(value):
                    return None

                return float(value)

            participant_rows.append(
                {
                    "study_id": question["study_id"],
                    "set_id": set_id,
                    "presentation_order":
                        question["presentation_order"],
                    "participant_id": participant_id,
                    "participant_rating": int(
                        rating_match.group(1)
                    ),
                    "participant_rating_label": str(answer),
                    "response_id": response_id,
                    "session_id": session_id,
                    "duration_seconds": duration,
                    "first_click_seconds": timing_value(1),
                    "last_click_seconds": timing_value(2),
                    "page_submit_seconds": timing_value(3),
                    "click_count": timing_value(4),
                }
            )

        if len(participant_rows) != 40:
            raise RuntimeError(
                f"Set {set_id:02d} produced "
                f"{len(participant_rows)} ratings instead of 40"
            )

        rows.extend(participant_rows)

    ratings = pd.DataFrame(rows)

    if len(ratings) != 600:
        raise RuntimeError(
            f"Expected 600 participant ratings, found {len(ratings)}"
        )

    ratings_per_sample = ratings.groupby("study_id").size()

    if len(ratings_per_sample) != 200:
        raise RuntimeError(
            f"Expected 200 unique samples, found "
            f"{len(ratings_per_sample)}"
        )

    if not ratings_per_sample.eq(3).all():
        invalid = ratings_per_sample[
            ratings_per_sample != 3
        ]

        raise RuntimeError(
            "Some samples do not have exactly three ratings:\n"
            f"{invalid}"
        )

    return ratings


def build_wide_table(samples, ratings):
    participant_columns = []

    grouped_rows = []

    for study_id, group in ratings.groupby(
        "study_id",
        sort=False,
    ):
        group = group.sort_values(
            ["set_id", "participant_id"]
        ).reset_index(drop=True)

        if len(group) != 3:
            raise RuntimeError(
                f"{study_id} has {len(group)} ratings instead of 3"
            )

        row = {"study_id": study_id}

        for position, participant in group.iterrows():
            number = position + 1

            row[f"participant_{number}_id"] = (
                participant["participant_id"]
            )
            row[f"participant_{number}_rating"] = int(
                participant["participant_rating"]
            )
            row[f"participant_{number}_set"] = int(
                participant["set_id"]
            )

            participant_columns.extend(
                [
                    f"participant_{number}_id",
                    f"participant_{number}_rating",
                    f"participant_{number}_set",
                ]
            )

        grouped_rows.append(row)

    wide_ratings = pd.DataFrame(grouped_rows)

    wide = samples.merge(
        wide_ratings,
        on="study_id",
        how="left",
        validate="one_to_one",
    )

    required_ratings = [
        "participant_1_rating",
        "participant_2_rating",
        "participant_3_rating",
    ]

    if wide[required_ratings].isna().any().any():
        missing = wide.loc[
            wide[required_ratings].isna().any(axis=1),
            "study_id",
        ].tolist()

        raise RuntimeError(
            f"Missing participant ratings for: {missing}"
        )

    return wide


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qualtrics",
        required=True,
        help="Path to the Qualtrics Excel export.",
    )
    parser.add_argument(
        "--pool",
        default="prolific_study/pool_200_manifest.csv",
        help="Path to pool_200_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="prolific_study/results",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(args.pool)

    expected_columns = {
        "study_id",
        "dataset",
        "system",
        "dataset_index",
        "reference",
        "hypothesis",
        "judge_severity",
        "source_file",
    }

    missing_columns = expected_columns - set(samples.columns)

    if missing_columns:
        raise RuntimeError(
            f"Pool file is missing columns: {missing_columns}"
        )

    if len(samples) != 200:
        raise RuntimeError(
            f"Expected 200 samples, found {len(samples)}"
        )

    if samples["study_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate study IDs found in the sample pool"
        )

    ratings = extract_participant_ratings(
        args.qualtrics
    )

    long_table = ratings.merge(
        samples,
        on="study_id",
        how="left",
        validate="many_to_one",
    )

    wide_table = build_wide_table(
        samples,
        ratings,
    )

    samples_path = (
        output_dir
        / "samples_with_judge_ratings.csv"
    )
    long_path = (
        output_dir
        / "participant_ratings_long.csv"
    )
    wide_path = (
        output_dir
        / "judge_and_participant_ratings.csv"
    )
    excel_path = (
        output_dir
        / "prolific_ratings_complete.xlsx"
    )

    samples.to_csv(samples_path, index=False)
    long_table.to_csv(long_path, index=False)
    wide_table.to_csv(wide_path, index=False)

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        samples.to_excel(
            writer,
            sheet_name="Samples",
            index=False,
        )
        long_table.to_excel(
            writer,
            sheet_name="Ratings Long",
            index=False,
        )
        wide_table.to_excel(
            writer,
            sheet_name="Ratings Wide",
            index=False,
        )

    print("\nExtraction complete")
    print(f"Samples: {len(samples)}")
    print(f"Participant ratings: {len(ratings)}")
    print(
        "Ratings per sample:",
        ratings.groupby("study_id").size().unique().tolist(),
    )
    print(f"\nSaved: {samples_path}")
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")
    print(f"Saved: {excel_path}")


if __name__ == "__main__":
    main()
