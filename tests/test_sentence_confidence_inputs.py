import copy
import json
from pathlib import Path
import tempfile
import unittest

from asr_pipeline.confidence import inputs
from asr_pipeline.confidence.paths import SentenceConfidenceFiles


class FourModelInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="asr-four-source-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = SentenceConfidenceFiles(
            self.root / "four-model-output",
            self.root / "paper",
            fusion_dir=self.root / "fusion",
        )
        self.sample = {
            "dataset_index": 3,
            "hyp": "Fused text.",
            "source_hyps": {model: f"{model} text." for model in inputs.SOURCE_MODELS},
        }
        self.data = {
            "dataset": "commonvoice",
            "split": "test",
            "asr_models": list(inputs.SOURCE_MODELS),
            "samples": [self.sample],
        }
        self.write_fusion()

    def write_fusion(self):
        path = inputs.fusion_result_path("commonvoice", "test", paths=self.paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data))

    def write_segments(self):
        path = (
            self.paths.confidence_dir
            / "segments/sentence_segments_commonvoice_test.json"
        )
        path.parent.mkdir(parents=True)
        data = {
            "dataset": "commonvoice",
            "split": "test",
            "asr_models": list(inputs.SOURCE_MODELS),
            "fusion_input": str(
                inputs.fusion_result_path("commonvoice", "test", paths=self.paths)
            ),
            "transcripts": [{"dataset_index": 3, "hyp": "Fused text.", "segments": []}],
        }
        path.write_text(json.dumps(data))
        return path, data

    def test_four_sources_are_returned_without_creating_output(self):
        self.assertEqual(
            inputs.load_source_hypotheses("commonvoice", "test", paths=self.paths),
            {3: self.sample["source_hyps"]},
        )
        self.assertFalse(self.paths.confidence_dir.exists())

    def test_fifth_model_metadata_is_rejected(self):
        self.data["asr_models"].append("whisper_ft_chunked")
        self.write_fusion()
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            inputs.load_fusion_samples("commonvoice", "test", paths=self.paths)

    def test_fifth_source_hypothesis_is_rejected(self):
        self.sample["source_hyps"]["whisper_ft_chunked"] = "Extra source"
        self.write_fusion()
        with self.assertRaisesRegex(ValueError, "four source hypotheses"):
            inputs.load_source_hypotheses("commonvoice", "test", paths=self.paths)

    def test_duplicate_fusion_ids_are_rejected(self):
        self.data["samples"].append(copy.deepcopy(self.sample))
        self.write_fusion()
        with self.assertRaisesRegex(ValueError, "Duplicate fusion"):
            inputs.load_fusion_samples("commonvoice", "test", paths=self.paths)

    def test_word_confidence_must_match_the_used_source(self):
        path = self.paths.baseline_dir / inputs.BENCHMARK_FILES["commonvoice"]["qwen"]
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_index": 3,
                            "hyp": "Newer unrelated output",
                            "segments": [],
                        }
                    ]
                }
            )
        )
        with self.assertRaisesRegex(ValueError, "differs from fusion source"):
            inputs.load_word_confidence_samples(
                "commonvoice", "qwen", "test", paths=self.paths
            )
        path.write_text(
            json.dumps(
                {"samples": [{"sample_index": 3, "hyp": "qwen text.", "segments": []}]}
            )
        )
        self.assertEqual(
            len(
                inputs.load_word_confidence_samples(
                    "commonvoice", "qwen", "test", paths=self.paths
                )
            ),
            1,
        )

    def test_old_segment_metadata_is_rejected(self):
        path, data = self.write_segments()
        del data["fusion_input"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "does not match"):
            inputs.load_confidence_segments("commonvoice", "test", paths=self.paths)

    def test_segments_must_use_main_fusion_text(self):
        path, data = self.write_segments()
        self.assertEqual(
            len(
                inputs.load_confidence_segments(
                    "commonvoice", "test", paths=self.paths
                )["transcripts"]
            ),
            1,
        )
        data["transcripts"][0]["hyp"] = "Older five-model text"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "differs from the fusion"):
            inputs.load_confidence_segments("commonvoice", "test", paths=self.paths)

    def test_shetland_uses_only_full_set_input(self):
        self.assertEqual(
            inputs.fusion_result_path("shetland", "full", paths=self.paths),
            self.paths.fusion_path("shetland", "full"),
        )
        with self.assertRaisesRegex(ValueError, "requires split='full'"):
            inputs.fusion_result_path("shetland", "test", paths=self.paths)


if __name__ == "__main__":
    unittest.main()
