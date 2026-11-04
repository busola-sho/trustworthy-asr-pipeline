"""Guard against stale/copied outputs, unsafe paths and weakened comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper_results.reporting import generate_reports as run


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asr-reporting-test-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        inputs = root / "inputs"
        records = []
        for number in range(122):
            relative = (
                Path("01_model_combination/data/baselines") / f"input{number}.json"
            )
            path = inputs / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            records.append(
                {
                    "path": str(Path("paper_results") / relative),
                    "role": "curated_input",
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            )
        input_files = root / "input_files.json"
        input_files.write_text(json.dumps({"files": records}))
        splits = root / "splits.json"
        splits.write_text(
            json.dumps(
                {
                    "current_assignments": {
                        d: {
                            "dev": list(range(n // 2)),
                            "test": list(range(n // 2, n)),
                            "excluded_calibration": [],
                        }
                        for d, n in (
                            ("commonvoice", 680),
                            ("edacc", 198),
                            ("english_dialects", 2543),
                        )
                    }
                }
            )
        )
        self.args = argparse.Namespace(
            saved_results_dir=inputs,
            output_dir=root / "outputs",
            original_results_dir=None,
            input_files=input_files,
            splits_file=splits,
            reports=["tables"],
            police_individual=None,
            police_fusion=None,
            shetland_anchored=None,
        )

    def invoke(self):
        with patch.object(run, "parse_args", return_value=self.args):
            run.main()

    def test_noop_writer_cannot_count_as_reproduced(self):
        with patch(
            "paper_results.reporting.combination.build_paper_tables.main",
            return_value=None,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing generated outputs"):
                self.invoke()
        self.assertFalse((self.args.output_dir / "run-manifest.json").exists())

    def test_nonempty_output_is_preserved(self):
        self.args.output_dir.mkdir()
        sentinel = self.args.output_dir / "existing.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.invoke()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_output_inside_input_is_rejected(self):
        self.args.output_dir = self.args.saved_results_dir / "new-output"
        with self.assertRaisesRegex(ValueError, "overlaps protected"):
            self.invoke()
        self.assertFalse(self.args.output_dir.exists())

    def test_missing_input_fails_before_output_creation(self):
        next(self.args.saved_results_dir.rglob("*.json")).unlink()
        with self.assertRaisesRegex(
            FileNotFoundError, "Required saved input file is missing"
        ):
            self.invoke()
        self.assertFalse(self.args.output_dir.exists())

    def test_changed_input_fails_before_output_creation(self):
        next(self.args.saved_results_dir.rglob("*.json")).write_text(
            '{"changed": true}'
        )
        with self.assertRaisesRegex(ValueError, "Input hash differs"):
            self.invoke()
        self.assertFalse(self.args.output_dir.exists())

    def test_numeric_comparison_keeps_structure_and_labels(self):
        self.assertTrue(run.compare_numeric({"x": [1.0]}, {"x": [1.0 + 1e-14]}, []))
        self.assertFalse(run.compare_numeric({"x": [1.0]}, {"x": [1.001]}, []))
        self.assertFalse(run.compare_numeric({"id": "01"}, {"id": "1"}, []))
        self.assertFalse(run.compare_numeric({"x": True}, {"x": 1}, []))
        self.assertFalse(run.compare_numeric({"x": 1}, {"y": 1}, []))

    def test_paths_are_explicit_and_independent_between_runs(self):
        from paper_results.reporting.paths import PaperReportFiles

        first = PaperReportFiles(self.args.saved_results_dir, self.args.output_dir)
        second = PaperReportFiles(Path("/tmp/other-inputs"), Path("/tmp/other-outputs"))
        self.assertEqual(
            first.verbalized_dir,
            self.args.saved_results_dir / "02_sentence_confidence/data/verbalized",
        )
        self.assertNotEqual(first.verbalized_dir, second.verbalized_dir)
        self.assertEqual(
            first.confidence_tables,
            self.args.output_dir / "02_sentence_confidence/tables",
        )

    def test_reporting_environment_is_restored_after_failure(self):
        import os

        previous = dict(os.environ)
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            with run.reporting_environment(self.args.output_dir):
                self.assertEqual(os.environ["MPLBACKEND"], "Agg")
                raise RuntimeError("writer failed")
        self.assertEqual(dict(os.environ), previous)

    def test_report_job_accepts_only_newly_written_expected_outputs(self):
        from paper_results.reporting.paths import PaperReportFiles

        paths = PaperReportFiles(self.args.saved_results_dir, self.args.output_dir)
        (paths.output_dir / "logs").mkdir(parents=True)
        relative = "01_model_combination/tables/synthetic.txt"
        generated = []
        run.run_report_job(
            "synthetic",
            lambda: (paths.output_dir / relative).write_text("Synthetic fixture"),
            [relative],
            paths=paths,
            reference=None,
            generated=generated,
        )
        self.assertEqual(generated[0]["path"], relative)
        self.assertTrue(generated[0]["absent_before_job"])


if __name__ == "__main__":
    unittest.main()
