"""Safety checks for the user-facing single-clip command; no inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from asr_pipeline import run_pipeline
from asr_pipeline.confidence import inputs


class RunPipelineTests(unittest.TestCase):
    def test_refuses_existing_directory_before_service_access(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "clip.wav"
            audio.write_bytes(b"fixture")
            with self.assertRaises(FileExistsError):
                run_pipeline.main(
                    [
                        "--audio",
                        str(audio),
                        "--reference",
                        "hello",
                        "--output-dir",
                        folder,
                    ]
                )

    def test_recording_preserves_contradictory_response(self):
        class Response:
            def model_dump(self):
                return {
                    "message": {
                        "content": "Severity: 3\nJustification: preserves meaning"
                    }
                }

        class Service:
            def chat(self, **kwargs):
                return response

        response = Response()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            result = run_pipeline.LoggedModelClient(Service(), path).chat(
                model="judge", messages=[]
            )
            self.assertIs(result, response)
            self.assertEqual(
                json.loads(path.read_text())["response"], response.model_dump()
            )

    def test_explicit_clip_input_still_validates_four_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fusion.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "clip",
                        "split": "full",
                        "asr_models": ["qwen"],
                        "samples": [],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "four ASR models"):
                inputs.load_fusion_samples("clip", "full", fusion_file=path)

    def test_reference_is_optional_and_never_enters_audio_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            audio = parent / "test.wav"
            audio.write_bytes(b"synthetic audio")
            reference = parent / "reference.txt"
            reference.write_text("private evaluation reference")
            for name, reference_file in (("labelled", reference), ("unlabelled", None)):
                root = parent / name
                copied_audio, text = run_pipeline.load_input(
                    str(audio), reference_file, root
                )
                self.assertEqual(copied_audio.read_bytes(), audio.read_bytes())
                input_files = json.loads((root / "input/audio.json").read_text())
                self.assertEqual(input_files[0]["ref"], "")
                self.assertEqual(
                    text, reference.read_text() if reference_file else None
                )

    def test_direct_audio_url_uses_downloaded_bytes(self):
        import io

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            with patch.object(
                run_pipeline, "urlopen", return_value=io.BytesIO(b"audio response")
            ) as download:
                audio, reference = run_pipeline.load_input(
                    "https://example.test/clip.wav", None, root
                )
            download.assert_called_once_with(
                "https://example.test/clip.wav", timeout=60
            )
            self.assertEqual(audio.read_bytes(), b"audio response")
            self.assertIsNone(reference)

    def test_missing_reference_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "new"
            with self.assertRaises(FileNotFoundError):
                run_pipeline.load_input(
                    "unused.wav", Path(folder) / "missing.txt", root
                )
            self.assertFalse(root.exists())

    def test_no_reference_means_no_evaluation_calls_or_outputs(self):
        client = object()
        with tempfile.TemporaryDirectory() as folder:
            result = run_pipeline.evaluate_transcripts(
                {}, None, Path(folder), client, []
            )
            self.assertEqual(result, {})
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_segmentation_uses_hypothesis_only(self):
        from types import SimpleNamespace
        from asr_pipeline.config import SOURCE_MODELS

        samples = [{"dataset_index": 0, "hyp": "We went home.", "ref": "SECRET"}]
        splitter = SimpleNamespace(split=lambda text: [text])
        segments = run_pipeline.segment_transcripts(
            samples, splitter, Path("fusion.json")
        )
        self.assertNotIn("SECRET", json.dumps(segments))
        self.assertIsNone(segments["transcripts"][0]["segments"][0]["severity"])
        self.assertEqual(segments["asr_models"], list(SOURCE_MODELS))

    def test_reference_evaluation_preserves_raw_judgements_and_fusion_inputs(self):
        import sys
        from types import SimpleNamespace

        class Splitter:
            def __init__(self, name):
                pass

            def split(self, text):
                return [text]

        class Judge:
            def __init__(self):
                self.requests = []

            def chat(self, **request):
                self.requests.append(request)
                return SimpleNamespace(
                    message=SimpleNamespace(
                        content="Severity: 3\nJustification: preserves meaning"
                    )
                )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fused = root / "fusion.json"
            data = {
                "dataset": "clip",
                "split": "full",
                "samples": [{"dataset_index": 0, "ref": "", "hyp": "We went home."}],
            }
            fused.write_text(json.dumps(data))
            original_bytes = fused.read_bytes()
            judge = Judge()
            with patch.dict(
                sys.modules,
                {
                    "wtpsplit": SimpleNamespace(SaT=Splitter),
                    "ollama": SimpleNamespace(Client=None),
                },
            ):
                evaluated = run_pipeline.evaluate_transcripts(
                    {"unanchored": fused}, "Evaluation reference only.", root, judge, []
                )
            self.assertEqual(fused.read_bytes(), original_bytes)
            output = json.loads(evaluated["unanchored"].read_text())
            self.assertEqual(output["samples"][0]["severity"], 3)
            self.assertIn(
                "preserves meaning", output["samples"][0]["judge_raw_response"]
            )
            segments = json.loads(
                (
                    root / "evaluation/segments/sentence_segments_clip_full.json"
                ).read_text()
            )
            self.assertEqual(segments["num_segments_parsed"], 1)
            self.assertEqual(segments["transcripts"][0]["segments"][0]["severity"], 3)
            self.assertEqual(len(judge.requests), 2)


if __name__ == "__main__":
    unittest.main()
