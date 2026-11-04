import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import List, Optional
import unittest
from unittest.mock import patch

from asr_pipeline.config import SOURCE_MODELS, PROJECT_ROOT
from asr_pipeline import evaluation, fusion
from asr_pipeline.files import read_transcript_files, check_new_output_path
from asr_pipeline.confidence.paths import SentenceConfidenceFiles


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content=self.response))


class ReleasePipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="asr-release-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.hypotheses = dict(
            zip(
                SOURCE_MODELS,
                ["we went home", "we went home", "we went home", "we went away"],
            )
        )

    def input_files(self):
        paths = {}
        for model in SOURCE_MODELS:
            path = self.root / f"{model}.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "dataset_index": 3,
                                "ref": "we went home",
                                "hyp": self.hypotheses[model],
                                "segments": [],
                            }
                        ]
                    }
                )
            )
            paths[model] = path.name
        input_files = self.root / "sources.json"
        input_files.write_text(
            json.dumps({"dataset": "commonvoice", "split": "test", "sources": paths})
        )
        return input_files

    def test_rover_and_mbr(self):
        self.assertEqual(fusion.combine(self.hypotheses, "rover")[0], "we went home")
        text, metadata = fusion.combine(self.hypotheses, "mbr")
        self.assertEqual(text, "we went home")
        self.assertEqual(metadata["selected_model"], "qwen")

    def test_selection_returns_an_existing_candidate(self):
        client = FakeClient("B")
        text, metadata = fusion.combine(
            self.hypotheses, "selection", client=client, index=7
        )
        self.assertEqual(text, self.hypotheses[metadata["model_order"][1]])
        self.assertIn(text, self.hypotheses.values())
        with self.assertRaises(ValueError):
            fusion.combine(
                self.hypotheses, "selection", client=FakeClient("no valid label")
            )

    def test_anchor_and_unanchored_keep_their_prompts(self):
        for method in ("anchored", "unanchored"):
            client = FakeClient("we went home")
            self.assertEqual(
                fusion.combine(self.hypotheses, method, client=client)[0],
                "we went home",
            )
            self.assertEqual(len(client.calls), 1)
        client = FakeClient("ok")
        fusion.combine(self.hypotheses, "anchored", client=client)
        self.assertIn("TRANSCRIPT A (base", client.calls[0]["messages"][0]["content"])

    def test_fusion_writes_only_new_outputs_and_keeps_sources(self):
        input_files = self.input_files()
        before = {
            p: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.root.glob("*.json")
        }
        target = self.root / "fusion.json"
        fusion.combine_transcript_files(input_files, target, "rover")
        data = json.loads(target.read_text())
        self.assertEqual(set(data["samples"][0]["source_segments"]), set(SOURCE_MODELS))
        for path, file_checksum in before.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), file_checksum
            )
        with self.assertRaises(FileExistsError):
            fusion.combine_transcript_files(input_files, target, "rover")

    def test_transcript_files_require_matching_sample_indices(self):
        input_files = self.input_files()
        path = self.root / "qwen.json"
        path.write_text(json.dumps({"samples": []}))
        with self.assertRaisesRegex(ValueError, "same sample indices"):
            read_transcript_files(input_files)

    def test_severity_policy_without_model_calls(self):
        client = FakeClient("Severity: 2")
        self.assertEqual(
            evaluation.evaluate_sample({"ref": "hello", "hyp": ""}, client, "fake")[
                "severity"
            ],
            4,
        )
        self.assertIsNone(
            evaluation.evaluate_sample(
                {"ref": "<NOISE>", "hyp": "hello"}, client, "fake"
            )["severity"]
        )
        self.assertEqual(
            evaluation.evaluate_sample(
                {"ref": "hello", "hyp": "hey", "severity": 1}, client, "fake"
            )["severity"],
            1,
        )
        self.assertEqual(client.calls, [])

    def test_synthetic_severity_output_preserves_input(self):
        path = self.root / "input.json"
        path.write_text(
            json.dumps({"samples": [{"ref": "we went home", "hyp": "we went away"}]})
        )
        original = path.read_bytes()
        target = self.root / "scored.json"
        evaluation.evaluate_transcript_file(path, target, FakeClient("Severity: 3"))
        self.assertEqual(json.loads(target.read_text())["mean_severity"], 3)
        self.assertEqual(path.read_bytes(), original)

    def test_preserved_result_directories_are_protected(self):
        for directory in ("paper_results", "_archive", ".git"):
            with self.assertRaises(ValueError):
                check_new_output_path(PROJECT_ROOT / directory / "never-write.json")
        with self.assertRaises(ValueError):
            SentenceConfidenceFiles(
                PROJECT_ROOT / "paper_results/new", PROJECT_ROOT / "paper_results"
            )

    def test_asr_record_dataclasses_survive_adapter_extraction(self):
        path = PROJECT_ROOT / "asr_pipeline/asr_models.py"
        tree = ast.parse(path.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef)
            and n.name in {"WordConfidence", "ASRTranscript"}
        ]
        scope = {"dataclass": dataclass, "Optional": Optional, "List": List}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        self.assertEqual(scope["WordConfidence"]("word", None, None, 0.5).word, "word")

    def test_transcription_entrypoint_with_synthetic_audio_adapter(self):
        from asr_pipeline.transcription import transcribe_audio_files as run

        @dataclass
        class Word:
            word: str
            confidence: float

        audio = self.root / "synthetic.wav"
        audio.write_bytes(b"synthetic fixture; decoding is mocked")
        input_files = self.root / "audio.json"
        input_files.write_text(
            json.dumps([{"audio": audio.name, "dataset_index": 4, "ref": "hello"}])
        )
        asr_model = SimpleNamespace(
            transcribe=lambda data, sample_rate: SimpleNamespace(
                text="hello", segments=[Word("hello", 0.9)]
            )
        )

        def load_audio(path, *, sr, mono):
            return [0.0], sr

        fake_librosa = SimpleNamespace(load=load_audio)
        target = self.root / "asr.json"
        with patch.dict(sys.modules, {"librosa": fake_librosa}):
            run("qwen", input_files, target, "synthetic", asr_model=asr_model)
        self.assertEqual(
            json.loads(target.read_text())["samples"][0]["segments"][0]["confidence"],
            0.9,
        )

    def test_confidence_entrypoint_passes_explicit_paths_and_arguments(self):
        from asr_pipeline.confidence import cli

        confidence_dir = self.root / "work"
        with patch.object(cli, "run_stage") as run_stage:
            cli.main(
                [
                    "agreement",
                    "--work-dir",
                    str(confidence_dir),
                    "--dataset",
                    "commonvoice",
                    "--split",
                    "test",
                ]
            )
        stage, arguments, paths = run_stage.call_args.args
        self.assertEqual(stage, "agreement")
        self.assertEqual(arguments, ["--dataset", "commonvoice", "--split", "test"])
        self.assertEqual(paths.confidence_dir, confidence_dir.resolve())
        self.assertFalse(confidence_dir.exists())

    def test_cli_help_never_imports_inference_dependencies(self):
        script = """
import sys
from asr_pipeline.__main__ import main
try:
    main(["--help"])
except SystemExit as e:
    assert e.code == 0
assert not any(x in sys.modules for x in ["torch", "ollama", "transformers", "wtpsplit"])
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
