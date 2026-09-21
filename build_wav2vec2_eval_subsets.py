"""Create Wav2Vec2 eval-suite subsets aligned to the existing Qwen subsets."""

import argparse
import json
from pathlib import Path

from jiwer import wer

from src.judge import normalise
from src.selector import find_canonical_file, load_samples


DATASETS = ["commonvoice", "edacc", "english_dialects"]
SUBSETS_DIR = Path("results/benchmarks/subsets")


def build_subset(dataset: str, overwrite: bool = False) -> None:
    manifest_path = SUBSETS_DIR / f"qwen3asr_{dataset}_sub150.json"
    output_path = SUBSETS_DIR / f"wav2vec2_{dataset}_sub150.json"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; inspect it or rerun with --overwrite"
        )

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    indices = manifest.get("subset_indices")
    manifest_samples = manifest.get("samples", [])
    if not indices or not manifest_samples:
        raise ValueError(f"Missing subset indices or samples: {manifest_path}")

    manifest_by_index = {
        sample["sample_index"]: sample
        for sample in manifest_samples
        if sample.get("sample_index") is not None
    }

    canonical_path = Path(find_canonical_file("wav2vec2", dataset))
    canonical_samples = load_samples(canonical_path)
    canonical_by_index = {
        sample["sample_index"]: sample
        for sample in canonical_samples
        if sample.get("sample_index") is not None
    }

    missing = [index for index in indices if index not in canonical_by_index]
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} requested indices are absent from "
            f"{canonical_path}: {missing[:10]}"
        )

    selected = []
    for index in indices:
        sample = dict(canonical_by_index[index])
        expected = manifest_by_index.get(index)
        if expected is None:
            raise ValueError(f"{dataset}: index {index} missing from Qwen samples")

        if normalise(sample.get("ref", "")) != normalise(expected.get("ref", "")):
            raise ValueError(
                f"{dataset}: reference mismatch at sample_index={index}"
            )

        selected.append(sample)

    refs = [normalise(sample.get("ref") or "") for sample in selected]
    hyps = [normalise(sample.get("hyp") or "") for sample in selected]

    output = {
        "model": "wav2vec2",
        "dataset": dataset,
        "num_samples": len(selected),
        "subset_indices": indices,
        "corpus_wer": wer(refs, hyps),
        "source_file": str(canonical_path),
        "subset_manifest": str(manifest_path),
        "samples": selected,
    }

    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"{dataset}: wrote {len(selected)} aligned samples to {output_path}")
    print(f"  source: {canonical_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for dataset in DATASETS:
        build_subset(dataset, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
