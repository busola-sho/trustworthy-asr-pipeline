"""Synthetic prediction checks: reference labels cannot affect confidence signals."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from asr_pipeline.config import SOURCE_MODELS
from asr_pipeline.run_pipeline import estimate_confidence


class ReferenceFreeConfidenceTests(unittest.TestCase):
    def test_all_three_signals_score_unlabelled_sentences_without_reference_leakage(
        self,
    ):
        class SentenceSplitter:
            def __init__(self, name):
                pass

            def split(self, text):
                return [text]

        class Client:
            def __init__(self):
                self.requests = []

            def chat(self, **request):
                self.requests.append(request)
                return SimpleNamespace(message=SimpleNamespace(content="Score: 75"))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fusion = root / "fusion.json"
            text = "We went home."
            sample = {
                "dataset_index": 0,
                "ref": "SECRET REFERENCE MUST NEVER REACH PREDICTION",
                "hyp": text,
                "source_hyps": dict.fromkeys(SOURCE_MODELS, text),
                "source_segments": {
                    model: [{"word": word, "confidence": 0.8} for word in text.split()]
                    for model in SOURCE_MODELS
                },
            }
            fusion.write_text(
                json.dumps(
                    {
                        "dataset": "clip",
                        "split": "full",
                        "asr_models": list(SOURCE_MODELS),
                        "samples": [sample],
                    }
                )
            )
            client = Client()
            # Only dependencies are substituted; all segmentation, alignment and scoring code runs.
            with patch.dict(
                sys.modules,
                {
                    "wtpsplit": SimpleNamespace(SaT=SentenceSplitter),
                    "ollama": SimpleNamespace(Client=None),
                },
            ):
                estimate_confidence(fusion, root, client, [])
            verbalized = json.loads(
                (
                    root
                    / "confidence/verbalized_confidence/verbalized_confscore_clip_full.json"
                ).read_text()
            )
            agreement = json.loads(
                (
                    root
                    / "confidence/cross_model_agreement/cross_model_agreement_clip_full.json"
                ).read_text()
            )
            internal = json.loads(
                (root / "confidence/model_internal_raw.json").read_text()
            )
            self.assertEqual(verbalized["num_segments_scored"], 1)
            self.assertEqual(
                verbalized["transcripts"][0]["segments"][0]["verbalized_score"], 75
            )
            self.assertEqual(
                agreement["transcripts"][0]["segments"][0]["crossmodel_mean"], 1.0
            )
            confidences = internal["transcripts"][0]["segments"][0][
                "per_model_confidence"
            ]
            for value in confidences.values():
                self.assertAlmostEqual(value, 0.8)
            self.assertNotIn("SECRET REFERENCE", json.dumps(client.requests))
            for output in (verbalized, agreement, internal):
                self.assertIsNone(output["transcripts"][0]["segments"][0]["severity"])

    def test_unlabelled_and_labelled_agreement_have_identical_calculations(self):
        from asr_pipeline.confidence.cross_model_agreement import run_dataset

        sources = {
            0: dict(
                zip(
                    SOURCE_MODELS,
                    ["we went home", "we went away", "we went home", "we stayed home"],
                )
            )
        }
        segments = {
            "transcripts": [
                {
                    "dataset_index": 0,
                    "segments": [{"segment": "we went home", "severity": None}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as folder:
            unlabelled = run_dataset(
                "clip",
                "full",
                segments_data=segments,
                source_hypotheses=sources,
                output_dir=Path(folder) / "unlabelled",
                include_unlabelled=True,
            )
            labelled_segments = copy.deepcopy(segments)
            labelled_segments["transcripts"][0]["segments"][0]["severity"] = 2
            labelled = run_dataset(
                "clip",
                "full",
                segments_data=labelled_segments,
                source_hypotheses=sources,
                output_dir=Path(folder) / "labelled",
            )
        first = unlabelled["transcripts"][0]["segments"][0]
        second = labelled["transcripts"][0]["segments"][0]
        for field in (
            "crossmodel_mean",
            "crossmodel_min",
            "local_reconstructions",
            "per_model_alignment_score",
        ):
            self.assertEqual(first[field], second[field])
