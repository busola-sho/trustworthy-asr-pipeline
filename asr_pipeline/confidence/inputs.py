"""Four-source sentence confidence: inputs."""

import json
from pathlib import Path
from asr_pipeline.config import SOURCE_MODELS
from .paths import SentenceConfidenceFiles

BENCHMARK_FILES = {
    "commonvoice": {
        "qwen": "qwen_commonvoice_20260722_merged.json",
        "whisperx": "whisperx_commonvoice_20260720_162322.json",
        "parakeet": "parakeet_commonvoice_merged.json",
        "wav2vec2": "wav2vec2_commonvoice_merged.json",
    },
    "edacc": {
        "qwen": "qwen_edacc_20260722_merged.json",
        "whisperx": "whisperx_edacc_20260719_025227.json",
        "parakeet": "parakeet_edacc_20260719_033319.json",
        "wav2vec2": "wav2vec2_edacc_20260719_033606.json",
    },
    "english_dialects": {
        "qwen": "qwen_english_dialects_20260722_merged.json",
        "whisperx": "whisperx_english_dialects_20260720_234027.json",
        "parakeet": "parakeet_english_dialects_merged.json",
        "wav2vec2": "wav2vec2_english_dialects_merged.json",
    },
}
SHETLAND_BENCHMARKS = {
    "qwen": "shetland_qwen3asr_20260603_150124.json",
    "whisperx": "shetland_whisper_20260603_123115.json",
    "parakeet": "shetland_parakeet_20260606_134131.json",
    "wav2vec2": "shetland_wav2vec2_20260606_134507.json",
}


def fusion_result_path(
    dataset, split, *, fusion_file=None, fusion_dir=None, paths=None
):
    if fusion_file is not None:
        return Path(fusion_file)
    paths = paths or SentenceConfidenceFiles(fusion_dir=fusion_dir)
    return paths.fusion_path(dataset, split)


def load_fusion_samples(
    dataset, split, *, fusion_file=None, fusion_dir=None, paths=None
):
    path = fusion_result_path(
        dataset, split, fusion_file=fusion_file, fusion_dir=fusion_dir, paths=paths
    )
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("dataset") != dataset or data.get("split") != split:
        raise ValueError(f"Fusion file has a different dataset or split: {path}")
    if set(data.get("asr_models", [])) != set(SOURCE_MODELS):
        raise ValueError(f"The file must name exactly the four ASR models: {path}")
    samples, seen = [], set()
    for sample in data["samples"]:
        if not sample.get("hyp") or sample.get("skipped") or sample.get("error"):
            continue
        index = sample["dataset_index"]
        if index in seen:
            raise ValueError(f"Duplicate fusion dataset_index={index}: {path}")
        seen.add(index)
        sources = sample.get("source_hyps", {})
        if set(sources) != set(SOURCE_MODELS):
            raise ValueError(f"Expected four source hypotheses at {index}: {path}")
        samples.append(
            {**sample, "source_hyps": {m: sources[m] or "" for m in SOURCE_MODELS}}
        )
    return samples


def load_source_hypotheses(dataset, split, *, fusion_file=None, paths=None):
    samples = load_fusion_samples(dataset, split, fusion_file=fusion_file, paths=paths)
    return {sample["dataset_index"]: sample["source_hyps"] for sample in samples}


def load_confidence_segments(
    dataset, split, *, confidence_dir=None, fusion_file=None, paths=None
):
    paths = paths or SentenceConfidenceFiles()
    confidence_dir = (
        Path(confidence_dir) if confidence_dir is not None else paths.confidence_dir
    )
    fusion_file = fusion_file or paths.fusion_path(dataset, split)
    path = paths.sentences_file or (
        confidence_dir / "segments" / f"sentence_segments_{dataset}_{split}.json"
    )
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    # Do not mix sentences from one fusion run with the scores from another.
    if (
        data.get("dataset") != dataset
        or data.get("split") != split
        or set(data.get("asr_models", [])) != set(SOURCE_MODELS)
        or data.get("fusion_input")
        != str(fusion_result_path(dataset, split, fusion_file=fusion_file))
    ):
        raise ValueError(
            f"The sentence file does not match the specified four-model fusion file: {path}"
        )
    fusion = {
        s["dataset_index"]: s["hyp"]
        for s in load_fusion_samples(dataset, split, fusion_file=fusion_file)
    }
    seen = set()
    for transcript in data["transcripts"]:
        index = transcript["dataset_index"]
        if (
            index in seen
            or index not in fusion
            or transcript.get("hyp") != fusion[index]
        ):
            raise ValueError(
                f"Sentence text differs from the fusion transcript at {index}: {path}"
            )
        seen.add(index)
    return data


def load_word_confidence_samples(
    dataset, model, split, *, fusion_file=None, paths=None
):
    if model not in SOURCE_MODELS:
        raise ValueError(f"Not a main four-source ASR model: {model}")
    fusion_samples = load_fusion_samples(
        dataset, split, fusion_file=fusion_file, paths=paths
    )
    sources = {s["dataset_index"]: s["source_hyps"] for s in fusion_samples}
    if fusion_samples and all("source_segments" in s for s in fusion_samples):
        return [
            {
                "sample_index": s["dataset_index"],
                "hyp": s["source_hyps"][model],
                "segments": s["source_segments"][model],
            }
            for s in fusion_samples
        ]
    if any("source_segments" in s for s in fusion_samples):
        raise ValueError("Mixed embedded and external word-confidence sources")
    paths = paths or SentenceConfidenceFiles()
    shetland_dir = paths.shetland_dir
    baseline_dir = paths.baseline_dir
    path = (
        shetland_dir / SHETLAND_BENCHMARKS[model]
        if dataset == "shetland"
        else baseline_dir / BENCHMARK_FILES[dataset][model]
    )
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    indexed = {}
    for position, sample in enumerate(data["samples"]):
        index = sample.get("sample_index")
        if index is None:
            index = sample.get("dataset_index")
        if index is None:
            index = position
        if index in sources:
            if index in indexed:
                raise ValueError(f"Duplicate benchmark index={index}: {path}")
            if " ".join((sample.get("hyp") or "").split()) != " ".join(
                sources[index][model].split()
            ):
                raise ValueError(
                    f"Benchmark hypothesis differs from fusion source at {index}: {path}"
                )
            indexed[index] = {**sample, "sample_index": index}
    if indexed.keys() != sources.keys():
        raise ValueError(f"Benchmark lacks fusion source IDs: {path}")
    return list(indexed.values())
