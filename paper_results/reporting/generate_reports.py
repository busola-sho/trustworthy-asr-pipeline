#!/usr/bin/env python3
"""Make tables and figures from saved paper results. Do not run ASR or LLMs."""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
COMBINATION = "01_model_combination"
CONFIDENCE = "02_sentence_confidence"
from paper_results.reporting.common import load_split_assignments, get_indices_for_split
from paper_results.reporting.paths import PaperReportFiles
from asr_pipeline.runtime import temporary_environment


def file_checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_numeric(actual, reference, differences, numeric_strings=False):
    """Compare text and structure exactly. Allow numeric differences up to 1e-12."""
    if type(actual) is bool or type(reference) is bool:
        return type(actual) is type(reference) and actual == reference
    if isinstance(actual, dict) and isinstance(reference, dict):
        return actual.keys() == reference.keys() and all(
            compare_numeric(actual[k], reference[k], differences, numeric_strings)
            for k in actual
        )
    if isinstance(actual, list) and isinstance(reference, list):
        return len(actual) == len(reference) and all(
            compare_numeric(a, b, differences, numeric_strings)
            for a, b in zip(actual, reference)
        )
    if actual == reference:
        return True
    if type(actual) is int and type(reference) is int:
        return False
    numeric = isinstance(actual, (int, float)) and isinstance(reference, (int, float))
    csv_numbers = (
        numeric_strings and isinstance(actual, str) and isinstance(reference, str)
    )
    if numeric or csv_numbers:
        try:
            a, b = float(actual), float(reference)
        except (ValueError, TypeError):
            return False
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        differences.append(abs(a - b))
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return False


def reference_comparison(path, saved):
    actual, original = path.read_bytes(), saved.read_bytes()
    if "diagnostics/full_grid_" in str(path):
        actual = b"\n".join(actual.splitlines()[1:])
        original = b"\n".join(original.splitlines()[1:])
    if actual == original:
        return {
            "reference_comparison": "exact_except_grid_source_header"
            if "diagnostics/" in str(path)
            else "exact"
        }
    if path.suffix == ".csv":
        a, b = (
            list(csv.reader(io.StringIO(v.decode("utf-8")))) for v in (actual, original)
        )
        if not a or not b or a[0] != b[0]:
            raise RuntimeError(f"CSV header differs from saved reference: {path}")
    elif path.suffix == ".json":
        a, b = json.loads(actual), json.loads(original)
    else:
        raise RuntimeError(f"Generated report differs from saved reference: {path}")
    differences = []
    if not compare_numeric(a, b, differences, numeric_strings=path.suffix == ".csv"):
        raise RuntimeError(
            f"Generated report differs from saved reference beyond 1e-12: {path}"
        )
    return {
        "reference_comparison": "numeric_within_1e-12",
        "maximum_absolute_numeric_difference": max(differences, default=0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--saved-results-dir",
        "--input-root",
        type=Path,
        required=True,
        help="folder containing the saved paper inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty folder for generated tables and figures",
    )
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument(
        "--input-files",
        "--manifest",
        type=Path,
        required=True,
        help="JSON list of required input files and their checksums",
    )
    parser.add_argument(
        "--original-results-dir",
        "--reference-root",
        type=Path,
        help="optional folder of original outputs to compare against",
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        choices=("tables", "confidence-figures", "downstream"),
        default=["tables"],
    )
    parser.add_argument("--police-individual", type=Path)
    parser.add_argument("--police-fusion", type=Path)
    parser.add_argument("--shetland-anchored", type=Path)
    return parser.parse_args(argv)


def run_report_job(name, function, relative_paths, *, paths, reference, generated):
    outputs = paths.output_dir
    expected = [outputs / p for p in relative_paths]
    if any(p.exists() for p in expected):
        raise RuntimeError(f"Expected outputs already exist before job {name}")
    for p in expected:
        p.parent.mkdir(parents=True, exist_ok=True)
    before = {
        p
        for group in (COMBINATION, CONFIDENCE)
        for p in (outputs / group).rglob("*")
        if p.is_file()
    }
    started = time.time_ns()
    with (outputs / "logs" / f"{name}.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            function()
    after = {
        p
        for group in (COMBINATION, CONFIDENCE)
        for p in (outputs / group).rglob("*")
        if p.is_file()
    }
    if after - before != set(expected):
        missing = [
            str(p.relative_to(outputs)) for p in set(expected) - (after - before)
        ]
        unexpected = [
            str(p.relative_to(outputs)) for p in (after - before) - set(expected)
        ]
        raise RuntimeError(
            f"{name}: missing generated outputs {missing}; unexpected {unexpected}"
        )
    for path in expected:
        if not path.stat().st_size:
            raise RuntimeError(f"Empty generated output: {path}")
        item = {
            "path": str(path.relative_to(outputs)),
            "job": name,
            "absent_before_job": True,
            "sha256": file_checksum(path),
            "bytes": path.stat().st_size,
            "job_started_ns": started,
        }
        saved = reference / path.relative_to(outputs) if reference else None
        if saved and saved.is_file() and path.suffix in (".txt", ".csv", ".json"):
            item.update(reference_comparison(path, saved))
        else:
            item["reference_comparison"] = "not_compared"  # PNG/pickle/new outputs
        generated.append(item)
    print(f"Passed {name}: {len(expected)} newly generated outputs", flush=True)


def capture_output(path, function):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        function()
    path.write_text(buffer.getvalue(), encoding="utf-8")


def generate_tables(paths, reference, generated):
    inputs, outputs = paths.saved_results_dir, paths.output_dir
    combo_in, combo_out = inputs / COMBINATION, outputs / COMBINATION
    from .combination import build_paper_tables as strategy

    run_report_job(
        "strategy-guidance",
        lambda: strategy.main(paths=paths),
        [
            f"{COMBINATION}/tables/{name}.{ext}"
            for name in ("strategy_comparison", "prompt_guidance_ablation")
            for ext in ("csv", "txt")
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .combination import build_postedit_comparison as postedit

    run_report_job(
        "postedit",
        lambda: postedit.main(paths=paths),
        [
            f"{COMBINATION}/tables/single_hypothesis_postedit.{ext}"
            for ext in ("csv", "txt")
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .confidence import build_paper_confidence_table as confidence

    run_report_job(
        "confidence-summary",
        lambda: confidence.main(paths=paths),
        [f"{CONFIDENCE}/tables/confidence_summary.{ext}" for ext in ("csv", "txt")],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .confidence import plot_spearman_correlations as correlations

    run_report_job(
        "spearman",
        lambda: correlations.main(paths=paths),
        [
            f"{CONFIDENCE}/tables/spearman_correlations.csv",
            f"{CONFIDENCE}/figures/spearman_correlations_heatmap.png",
        ]
        + [
            f"{CONFIDENCE}/figures/by_dataset/spearman_{d}.png"
            for d in (
                "commonvoice",
                "edacc",
                "english_dialects",
                "shetland",
                "police_scotland",
            )
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .confidence import calibrate_verbalized_confidence as calibration

    run_report_job(
        "calibration-fit",
        lambda: calibration.main(paths=paths),
        [
            f"{CONFIDENCE}/calibration/{name}"
            for name in (
                "calibration_results.csv",
                "calibration_results.json",
                "verbalized_confscore_calibrator.pkl",
                "verbalized_confscore_reliability_calibrated.png",
            )
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .confidence import build_calibration_table as calibration_table

    run_report_job(
        "calibration-table",
        lambda: calibration_table.main(paths=paths),
        [
            f"{CONFIDENCE}/calibration/calibration_paper_table.{ext}"
            for ext in ("csv", "txt")
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )
    from .combination import build_leaderboard as leaderboard
    from .combination import build_grid_leaderboard as grid

    for split in ("dev", "test"):
        baseline_path = combo_out / "tables" / f"baselines_{split}.txt"
        run_report_job(
            f"baselines-{split}",
            lambda: capture_output(
                baseline_path,
                lambda: leaderboard.print_results(
                    leaderboard.collect_results(
                        [str(combo_in / "data/baselines")], split, paths=paths
                    ),
                    split,
                ),
            ),
            [str(baseline_path.relative_to(outputs))],
            paths=paths,
            reference=reference,
            generated=generated,
        )
        grid_path = combo_out / "tables/diagnostics" / f"full_grid_{split}.txt"

        def print_grid():
            data = grid.load_grid_data(str(combo_in / "data/grid"), split)
            print(f"Reading {split} grid results from: {combo_in / 'data/grid'}")
            grid.print_grid_table(data, split)
            grid.print_strategy_comparison(data, split)

        run_report_job(
            f"grid-{split}",
            lambda: capture_output(grid_path, print_grid),
            [str(grid_path.relative_to(outputs))],
            paths=paths,
            reference=reference,
            generated=generated,
        )


def generate_confidence_figures(paths, reference, generated):
    from .confidence import evaluate_saved_confidence as evaluation

    for variant, methods in (
        ("confscore", ("verbalized", "proxy_model")),
        ("probscore", tuple(evaluation.METHODS)),
    ):
        expected = []
        for method in methods:
            label = variant if method == "verbalized" else method
            filename = (
                f"boxplot_v2_{label}_{variant}.png"
                if method == "proxy_model"
                else f"boxplot_v2_{label}.png"
            )
            expected.append(f"{CONFIDENCE}/figures/{filename}")
            if variant == "confscore":
                stem = "confscore" if method == "verbalized" else "proxy_confscore"
                expected += [
                    f"{CONFIDENCE}/figures/by_dataset/{stem}_{d}.png"
                    for d in evaluation.DATASETS
                ]

        def plot():
            for method in methods:
                evaluation.plot_boxplot(
                    method, variant, paths.confidence_figures, paths=paths
                )

        run_report_job(
            f"boxplots-{variant}",
            plot,
            expected,
            paths=paths,
            reference=reference,
            generated=generated,
        )


def generate_downstream_figures(args, paths, reference, generated):
    inputs, outputs = paths.saved_results_dir, paths.output_dir
    combo_in, combo_out = inputs / COMBINATION, outputs / COMBINATION
    from .combination import plot_downstream_distributions as downstream

    shetland = combo_in / "data/shetland"
    distributions = {
        "Unanchored Fusion": downstream.distribution_from_result(
            shetland / "naive_shetland_gemma4sel_full.json"
        ),
        "Anchored Fusion": downstream.distribution_from_result(args.shetland_anchored),
        "MBR-Style Consensus (baseline)": downstream.distribution_from_result(
            shetland / "mbr_shetland_full.json"
        ),
        "ROVER (baseline)": downstream.distribution_from_result(
            shetland / "rover_shetland_full.json"
        ),
        "Best Single Model (qwen)": downstream.distribution_from_result(
            shetland / "shetland_qwen3asr_20260603_150124.json"
        ),
    }
    police = downstream.police_distributions(args.police_individual, args.police_fusion)

    def plot_downstream():
        for dataset, data, title, panel in (
            ("shetland", distributions, "Shetland", "Shetland (Held-out transfer set)"),
            (
                "police_scotland",
                police,
                "Police Scotland",
                "Operational Police Data - Police Scotland",
            ),
        ):
            downstream.plot_binary(
                data,
                f"Meaning Alteration Rate by Strategy, {title}",
                panel,
                combo_out / "figures" / f"{dataset}_binary.png",
            )
            downstream.plot_severity(
                data,
                f"Severity Distribution by Strategy, {title}",
                panel,
                combo_out / "figures" / f"{dataset}_severity.png",
            )

    run_report_job(
        "downstream",
        plot_downstream,
        [
            f"{COMBINATION}/figures/{d}_{kind}.png"
            for d in ("shetland", "police_scotland")
            for kind in ("binary", "severity")
        ],
        paths=paths,
        reference=reference,
        generated=generated,
    )


def validate_inputs(args):
    inputs, outputs = args.saved_results_dir.resolve(), args.output_dir.resolve()
    reference = (
        args.original_results_dir.resolve() if args.original_results_dir else None
    )
    # Never target originals, their ancestors, or a descendant of input data.
    for protected in (inputs, PACKAGE, reference):
        if protected and (
            outputs == protected
            or outputs.is_relative_to(protected)
            or protected.is_relative_to(outputs)
        ):
            raise ValueError(
                f"Output overlaps protected input/code/reference folder: {protected}"
            )
    if outputs.exists() and (not outputs.is_dir() or any(outputs.iterdir())):
        raise ValueError(
            "Choose a new or empty output folder. Existing files will not be replaced"
        )
    input_files = json.loads(args.input_files.read_text())
    records = [r for r in input_files["files"] if r["role"] == "curated_input"]
    if len(records) != 122 or len({r["path"] for r in records}) != 122:
        raise ValueError(
            "Expected 122 different paper input files in the input file list"
        )
    for record in records:
        path = inputs / Path(record["path"]).relative_to("paper_results")
        if not path.is_file():
            raise FileNotFoundError(f"Required saved input file is missing: {path}")
        if file_checksum(path) != record["sha256"]:
            raise ValueError(f"Input hash differs from the saved paper input: {path}")
    if "downstream" in args.reports:
        for option in ("police_individual", "police_fusion", "shetland_anchored"):
            value = getattr(args, option)
            if value is None or not value.is_file():
                raise ValueError(
                    f"downstream requires an existing --{option.replace('_', '-')}"
                )
    paths = PaperReportFiles(inputs, outputs, load_split_assignments(args.splits_file))
    for record in records:
        if "/data/grid/" not in record["path"]:
            continue
        path = inputs / Path(record["path"]).relative_to("paper_results")
        data = json.loads(path.read_text())
        actual = {s["dataset_index"] for s in data["samples"]}
        if actual != set(get_indices_for_split(data["dataset"], data["split"], paths)):
            raise ValueError(
                f"Frozen split membership differs from declared grid snapshot: {path}"
            )
    return paths, reference


def save_report_summary(args, paths, generated):
    inputs, outputs = paths.saved_results_dir, paths.output_dir
    extra_inputs = {
        str(path.resolve()): file_checksum(path)
        for path in (args.police_individual, args.police_fusion, args.shetland_anchored)
        if path is not None
    }
    code_paths = {
        Path(module.__file__).resolve()
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
        and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(PACKAGE)
    }
    data = {
        "input_root": str(inputs),
        "output_root": str(outputs),
        "splits_sha256": file_checksum(args.splits_file),
        "input_manifest_sha256": file_checksum(args.input_files),
        "explicit_extra_inputs_sha256": extra_inputs,
        "loaded_reporting_code_sha256": {
            str(p.relative_to(PACKAGE)): file_checksum(p) for p in sorted(code_paths)
        },
        "python_version": sys.version,
        "reports": args.reports,
        "generated_outputs": generated,
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ("jiwer", "numpy", "scipy", "scikit-learn", "matplotlib")
        },
    }
    (outputs / "run-manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"Verified {len(generated)} new output files; details: {outputs / 'run-manifest.json'}"
    )


def reporting_environment(output_dir):
    """Keep plotting caches in the output folder without changing the caller's settings."""
    return temporary_environment(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(output_dir / ".matplotlib"),
            "XDG_CACHE_HOME": str(output_dir / ".cache"),
        }
    )


def main(argv=None):
    args = parse_args(argv)
    paths, reference = validate_inputs(args)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "logs").mkdir()
    generated = []
    with reporting_environment(paths.output_dir):
        if "tables" in args.reports:
            generate_tables(paths, reference, generated)
        if "confidence-figures" in args.reports:
            generate_confidence_figures(paths, reference, generated)
        if "downstream" in args.reports:
            generate_downstream_figures(args, paths, reference, generated)
        save_report_summary(args, paths, generated)


if __name__ == "__main__":
    main()
