"""Transcribe audio files with one ASR model. Load weights only when called."""

from dataclasses import asdict
from pathlib import Path

from .config import SOURCE_MODELS
from .files import read_json, check_new_output_path, write_json


def load_asr_model(model):
    if model not in SOURCE_MODELS:
        raise ValueError(f"Unsupported ASR model: {model}")
    if model == "whisperx":
        return WhisperXTranscriber()
    from .asr_models import Qwen3ASR, Parakeet, Wav2Vec2

    asr_model = {"qwen": Qwen3ASR, "parakeet": Parakeet, "wav2vec2": Wav2Vec2}[model]()
    asr_model.load()
    return asr_model


class WhisperXTranscriber:
    def __init__(self):
        import torch
        import whisperx

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisperx.load_model(
            "large-v3",
            device=self.device,
            compute_type="float16" if self.device == "cuda" else "int8",
            language="en",
        )
        self.align_model, self.metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

    def transcribe(self, audio, sample_rate):
        import whisperx
        from .asr_models import WordConfidence, ASRTranscript

        result = self.model.transcribe(audio, batch_size=16)
        aligned = whisperx.align(
            result["segments"],
            self.align_model,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        words = [
            WordConfidence(
                word=w["word"],
                start=w.get("start"),
                end=w.get("end"),
                confidence=w.get("score"),
            )
            for w in aligned.get("word_segments", [])
            if w.get("word")
        ]
        return ASRTranscript(
            text=" ".join(s["text"].strip() for s in result["segments"]),
            segments=words,
            model_name="whisperx-large-v3",
        )


def transcribe_audio_files(
    model, audio_files, output, dataset, split="full", asr_model=None
):
    check_new_output_path(output)
    source = Path(audio_files).resolve()
    entries = read_json(source)
    if not isinstance(entries, list):
        raise ValueError(
            "Audio files JSON must contain a list of audio paths and dataset_index values"
        )
    ids = [row["dataset_index"] for row in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("The audio file list repeats a dataset_index")
    paths = [
        (
            Path(row["audio"])
            if Path(row["audio"]).is_absolute()
            else source.parent / row["audio"]
        )
        for row in entries
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    import librosa

    asr_model = asr_model or load_asr_model(model)
    samples = []
    for row, path in zip(entries, paths):
        audio, sample_rate = librosa.load(path, sr=16000, mono=True)
        result = asr_model.transcribe(audio, sample_rate)
        samples.append(
            {
                "dataset_index": row["dataset_index"],
                "ref": row.get("ref", ""),
                "hyp": result.text,
                "segments": [asdict(s) for s in result.segments],
            }
        )
    write_json(
        output, {"model": model, "dataset": dataset, "split": split, "samples": samples}
    )
