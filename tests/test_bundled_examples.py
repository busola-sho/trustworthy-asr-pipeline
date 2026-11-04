"""Bundled quickstart inputs work without corpus caches or audio dependencies."""

import csv
import hashlib
import json
from pathlib import Path
import unittest
import wave


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/scottish"


class BundledExamplesTests(unittest.TestCase):
    def test_five_native_wavs_and_original_utf8_references(self):
        input_files = json.loads(
            (EXAMPLES / "examples.json").read_text(encoding="utf-8")
        )
        rows = input_files["examples"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            sum(row["source_subset"] == "scottish_female" for row in rows), 3
        )
        self.assertEqual(
            sum(row["source_subset"] == "scottish_male" for row in rows), 2
        )
        self.assertEqual(len({row["speaker_id"] for row in rows}), 5)
        with (EXAMPLES / "source_references.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            source_index = {row["file_id"]: row for row in csv.DictReader(handle)}
        for row in rows:
            with self.subTest(audio=row["audio"]):
                audio = EXAMPLES / row["audio"]
                reference = EXAMPLES / row["reference"]
                self.assertEqual(
                    reference.read_text(encoding="utf-8"), row["text"] + "\n"
                )
                self.assertEqual(
                    source_index[row["source_file_id"]]["text"], row["text"]
                )
                self.assertEqual(
                    source_index[row["source_file_id"]]["line_id"], row["line_id"]
                )
                self.assertEqual(
                    hashlib.sha256(audio.read_bytes()).hexdigest(), row["audio_sha256"]
                )
                self.assertEqual(
                    hashlib.sha256(reference.read_bytes()).hexdigest(),
                    row["reference_sha256"],
                )
                with wave.open(str(audio)) as recording:
                    self.assertEqual(recording.getframerate(), 48000)
                    self.assertEqual(recording.getnchannels(), 1)
                    self.assertEqual(recording.getsampwidth(), 2)
                    self.assertEqual(recording.getnframes(), row["frames"])
                    self.assertAlmostEqual(
                        recording.getnframes() / recording.getframerate(),
                        row["duration_seconds"],
                    )
                    self.assertLess(
                        recording.getnframes() / recording.getframerate(), 5
                    )
        self.assertEqual(
            {p.name for p in EXAMPLES.glob("*.wav")}, {r["audio"] for r in rows}
        )
        self.assertEqual(
            {p.name for p in EXAMPLES.glob("*.txt")}, {r["reference"] for r in rows}
        )

    def test_data_licence_and_quickstart_paths_are_present(self):
        self.assertIn(
            "Attribution-ShareAlike 4.0 International",
            (EXAMPLES / "LICENSE").read_text(),
        )
        readme = (EXAMPLES.parents[1] / "README.md").read_text(encoding="utf-8")
        self.assertIn("examples/scottish/scm_07508_01418625966.wav", readme)
        self.assertIn("examples/scottish/scm_07508_01418625966.txt", readme)
        self.assertIn("examples/scottish/*.wav", readme)
        self.assertNotIn("runs/live-walkthrough-", readme)
        self.assertIn("run", readme)


if __name__ == "__main__":
    unittest.main()
