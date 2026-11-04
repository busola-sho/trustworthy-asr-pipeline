"""The two displays share scores, align sentences explicitly and retain raw outputs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from asr_pipeline.config import SOURCE_MODELS
from asr_pipeline.display_results import (
    build_results_summary,
    render_html_summary,
    render_terminal_summary,
)


class ResultsDisplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.text = "We <went> home & stayed."
        sample = {"dataset_index": 0, "ref": "", "hyp": self.text}
        for model in SOURCE_MODELS:
            self.write(f"asr/{model}.json", {"samples": [sample]})
        self.combined = {"unanchored": self.root / "fusion/unanchored.json"}
        self.write("fusion/unanchored.json", {"samples": [sample]})
        segment = {"segment": self.text, "severity": None}
        self.write_sentences(
            "confidence/segments/sentence_segments_clip_full.json", [segment]
        )
        self.write_sentences(
            "confidence/verbalized_confidence/verbalized_confscore_clip_full.json",
            [
                {**segment, "verbalized_score": 87.5},
            ],
        )
        self.write_sentences(
            "confidence/cross_model_agreement/cross_model_agreement_clip_full.json",
            [
                {**segment, "crossmodel_mean": 0.812345, "crossmodel_min": 0.612345},
            ],
        )
        self.summary = {"unavailable": {"calibrated_confidence": "parameters required"}}

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def write_sentences(self, relative, segments):
        return self.write(
            relative, {"transcripts": [{"dataset_index": 0, "segments": segments}]}
        )

    def test_terminal_and_html_show_same_reference_transcripts_and_sentence_metrics(
        self,
    ):
        reference = "We <went> home & stayed."
        evaluated = self.write(
            "evaluation/unanchored.json",
            {
                "samples": [
                    {
                        "hyp": self.text,
                        "severity": 3,
                        "judge_raw_response": "Severity: 3; preserves meaning",
                    }
                ],
                "corpus_wer": 0.25,
            },
        )
        self.write_sentences(
            "evaluation/segments/sentence_segments_clip_full.json",
            [
                {
                    "segment": self.text,
                    "severity": 2,
                    "corresponding_span": reference,
                    "raw_judge_response": "Severity: 2; preserves meaning",
                },
            ],
        )
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        results = build_results_summary(
            self.root, self.combined, {"unanchored": evaluated}, reference
        )
        terminal = render_terminal_summary(results, self.summary["unavailable"])
        page = render_html_summary(results, self.summary, Path("clip.wav"), self.root)
        for display in (terminal, page):
            for value in (
                "Qwen3-ASR",
                "WhisperX",
                "Parakeet",
                "Wav2Vec2",
                "87.5/100",
                "0.8123",
                "0.6123",
                "25.0%",
            ):
                self.assertIn(value, display)
        self.assertIn(reference, terminal)
        self.assertIn("We &lt;went&gt; home &amp; stayed.", page)
        self.assertNotIn("We <went>", page)
        self.assertEqual(results["sentences"][0]["severity"], 2)
        self.assertEqual(results["sentences"][0]["wer"], 0)
        self.assertEqual(results["combination"][0]["severity"], 3)
        self.assertIn("<details>", page)
        self.assertIn("preserves meaning", page)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_reference_free_display_keeps_signals_and_marks_evaluation_unavailable(
        self,
    ):
        results = build_results_summary(self.root, self.combined, {}, None)
        self.assertIsNone(results["reference"])
        self.assertIsNone(results["asr"][0]["wer"])
        self.assertIsNone(results["sentences"][0]["severity"])
        self.assertEqual(results["sentences"][0]["verbalized_score"], 87.5)
        self.assertIn("evaluation skipped", render_terminal_summary(results, {}))

    def test_sentence_text_mismatch_cannot_silently_attach_wrong_scores(self):
        self.write_sentences(
            "confidence/cross_model_agreement/cross_model_agreement_clip_full.json",
            [
                {"segment": "Another sentence.", "crossmodel_mean": 1.0},
            ],
        )
        with self.assertRaisesRegex(ValueError, "misaligned scores"):
            build_results_summary(self.root, self.combined, {}, None)

    def test_display_does_not_modify_shared_summary_data(self):
        results = build_results_summary(self.root, self.combined, {}, None)
        before = copy.deepcopy(results)
        render_terminal_summary(results, {})
        render_html_summary(results, self.summary, Path("clip.wav"), self.root)
        self.assertEqual(results, before)
