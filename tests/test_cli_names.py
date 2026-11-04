"""Clear option names keep the earlier spellings and external library names working."""

import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from asr_pipeline.__main__ import create_parser
from asr_pipeline.run_pipeline import parse_arguments, connect_to_ollama
from asr_pipeline.confidence import cli
from paper_results.reporting.generate_reports import (
    parse_args as parse_report_arguments,
)


class CommandNamesTests(unittest.TestCase):
    def test_audio_file_option_accepts_previous_spelling(self):
        parser = create_parser()
        common = ["transcribe", "--model", "qwen", "--dataset", "clip"]
        current = parser.parse_args(
            [*common, "--audio-files", "audio_files.json", "--output-file", "new.json"]
        )
        previous = parser.parse_args(
            [*common, "--audio-manifest", "audio_files.json", "--output", "new.json"]
        )
        self.assertEqual(vars(current), vars(previous))

    def test_pipeline_options_accept_previous_spellings(self):
        common = ["--audio", "clip.wav", "--output-dir", "runs/new"]
        current = parse_arguments(
            [
                *common,
                "--reference-file",
                "reference.txt",
                "--cached-models-only",
                "--ollama-url",
                "http://localhost:11434",
            ]
        )
        previous = parse_arguments(
            [
                *common,
                "--reference",
                "reference.txt",
                "--offline",
                "--host",
                "http://localhost:11434",
            ]
        )
        self.assertEqual(vars(current), vars(previous))

    def test_reporting_options_accept_previous_spellings(self):
        common = ["--output-dir", "reports", "--splits-file", "splits.json"]
        current = parse_report_arguments(
            [
                *common,
                "--saved-results-dir",
                "paper_results",
                "--input-files",
                "input_files.json",
                "--original-results-dir",
                "paper_results",
            ]
        )
        previous = parse_report_arguments(
            [
                *common,
                "--input-root",
                "paper_results",
                "--manifest",
                "input_files.json",
                "--reference-root",
                "paper_results",
            ]
        )
        self.assertEqual(vars(current), vars(previous))

    def test_sentence_inputs_are_passed_explicitly_to_the_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(cli, "run_stage") as stage:
                cli.main(
                    [
                        "agreement",
                        "--confidence-dir",
                        folder,
                        "--fusion-file",
                        "fusion.json",
                        "--sentences-file",
                        "sentences.json",
                        "--dataset",
                        "clip",
                        "--split",
                        "full",
                        "--include-unlabelled",
                    ]
                )
            name, arguments, files = stage.call_args.args
            self.assertEqual(name, "agreement")
            self.assertEqual(files.fusion_file, Path("fusion.json").resolve())
            self.assertEqual(files.sentences_file, Path("sentences.json").resolve())
            self.assertIn("--include-unlabelled", arguments)

    def test_ollama_models_field_keeps_its_external_library_name(self):
        service = SimpleNamespace(
            list=lambda: SimpleNamespace(
                models=[
                    SimpleNamespace(model="gemma4:12b"),
                    SimpleNamespace(model="phi4:14b"),
                ]
            )
        )
        with patch.dict(
            sys.modules, {"ollama": SimpleNamespace(Client=lambda **kwargs: service)}
        ):
            self.assertIs(connect_to_ollama("http://localhost:11434"), service)

    def test_qwen_chunking_keeps_the_library_sr_keyword(self):
        source = Path(__file__).resolve().parents[1] / "asr_pipeline/asr_models.py"
        calls = [
            node
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "split_audio_into_chunks"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIn("sr", {keyword.arg for keyword in calls[0].keywords})


if __name__ == "__main__":
    unittest.main()
