"""
Add severity scores to existing benchmark JSON files without rerunning ASR.

Evaluation policy:
  * skipped/tag-only references remain unscored;
  * processing errors remain unscored;
  * a non-empty scorable reference paired with an empty hypothesis receives
    severity 4 deterministically (complete transcription failure);
  * all other unscored pairs are sent to the locked Phi-4 judge.

Existing non-null severity scores are preserved. Results are updated in place.
Use --mirror-to-main only when an additional copy in
writeup_results/benchmarks/main is explicitly required.

Examples:
    python rerunning/add_severity_to_existing_concurrent.py \
      --files writeup_results/clean_mbr_consensus/mbr_edacc_dev.json

    python rerunning/add_severity_to_existing_concurrent.py \
      --recursive-dir writeup_results/clean_grid --empty-only
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from ollama import Client

from severity_judge_prompts import DIRECT_SEVERITY_PROMPT
from src.concurrent_ollama import run_concurrent
from src.judge import is_tag_only


OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"
MIRROR_OUTPUT_DIR = Path("writeup_results/benchmarks/main")
MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))
BATCH_SIZE = 50

EMPTY_HYPOTHESIS_SEVERITY = 4
EMPTY_HYPOTHESIS_REASON = (
    "Deterministic evaluation rule: a non-empty scorable reference paired "
    "with an empty hypothesis is a complete transcription failure."
)


def parse_severity(text: str):
    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def ollama_severity(client: Client, ref: str, hyp: str, retries: int = 2):
    prompt = DIRECT_SEVERITY_PROMPT.format(reference=ref, hypothesis=hyp)

    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
                think=False,
            )
            text = response.message.content
            severity = parse_severity(text)
            if severity is not None:
                return severity, text
            if attempt < retries:
                time.sleep(0.5)
        except Exception as exc:
            if attempt == retries:
                print(f"    ERROR (severity judge): {exc}")
                return None, None
            time.sleep(1.0)

    return None, None


def _chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _normalised_text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_unscoreable_reference(sample: dict) -> bool:
    ref = _normalised_text(sample.get("ref"))
    return (
        not ref
        or sample.get("skipped", False)
        or is_tag_only(ref)
    )


def _is_empty_system_output(sample: dict) -> bool:
    return (
        not sample.get("error", False)
        and not _is_unscoreable_reference(sample)
        and not _normalised_text(sample.get("hyp"))
    )


def _recompute_summary(data: dict) -> tuple[float | None, list[int]]:
    severities = [
        sample["severity"]
        for sample in data.get("samples", [])
        if sample.get("severity") is not None
        and not sample.get("skipped", False)
        and not sample.get("error", False)
    ]

    mean_severity = sum(severities) / len(severities) if severities else None
    data["judge"] = f"{JUDGE_MODEL} (direct prompt; empty hypotheses deterministically scored 4)"
    data["mean_severity"] = (
        round(mean_severity, 3) if mean_severity is not None else None
    )
    data["severity_distribution"] = {
        str(level): severities.count(level)
        for level in range(5)
    }
    data["empty_hypothesis_policy"] = {
        "severity": EMPTY_HYPOTHESIS_SEVERITY,
        "description": EMPTY_HYPOTHESIS_REASON,
    }
    return mean_severity, severities


def process_file(
    file_path: Path,
    client: Client | None,
    max_workers: int = MAX_WORKERS,
    batch_size: int = BATCH_SIZE,
    empty_only: bool = False,
    mirror_to_main: bool = False,
):
    print(f"\n-- {file_path} --")

    with file_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    samples = data.get("samples", [])
    if not samples:
        print("  No samples found, skipping.")
        return

    def save_progress():
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

        if mirror_to_main:
            MIRROR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            mirror_path = MIRROR_OUTPUT_DIR / file_path.name
            if mirror_path.resolve() != file_path.resolve():
                with mirror_path.open("w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, ensure_ascii=False)

    deterministic_count = 0
    work_items = []

    for index, sample in enumerate(samples):
        if sample.get("severity") is not None:
            continue

        if sample.get("error", False):
            continue

        if _is_unscoreable_reference(sample):
            sample["severity"] = None
            continue

        if _is_empty_system_output(sample):
            sample["severity"] = EMPTY_HYPOTHESIS_SEVERITY
            sample["severity_assignment"] = "deterministic_empty_hypothesis"
            sample["judge_raw_response"] = EMPTY_HYPOTHESIS_REASON
            deterministic_count += 1
            continue

        if not empty_only:
            work_items.append((index, sample["ref"], sample["hyp"]))

    print(f"  Empty hypotheses assigned severity 4: {deterministic_count}")

    if empty_only:
        print("  Empty-only mode: no judge calls will be made.")
    else:
        if client is None:
            raise RuntimeError("An Ollama client is required when not using --empty-only")

        print(
            f"  {len(work_items)} samples queued for judging "
            f"(batches of {batch_size}, {max_workers} workers)"
        )

        def worker(item):
            _, ref, hyp = item
            return ollama_severity(client, ref, hyp)

        start_time = time.time()
        judged = 0

        for batch_number, batch in enumerate(
            _chunked(work_items, batch_size), start=1
        ):
            results = run_concurrent(
                batch,
                worker,
                max_workers=max_workers,
                progress_every=0,
            )

            for (index, _, _), result in zip(batch, results):
                severity, raw_response = (
                    result if result is not None else (None, None)
                )
                samples[index]["severity"] = severity
                samples[index]["judge_raw_response"] = raw_response
                if severity is not None:
                    samples[index]["severity_assignment"] = "llm_judge"

            judged += len(batch)
            elapsed = time.time() - start_time
            print(
                f"  batch {batch_number}: {judged}/{len(work_items)} judged "
                f"({elapsed:.0f}s elapsed)"
            )
            _recompute_summary(data)
            save_progress()

    mean_severity, severities = _recompute_summary(data)
    save_progress()

    remaining = sum(
        sample.get("severity") is None
        and not sample.get("skipped", False)
        and not sample.get("error", False)
        and not _is_unscoreable_reference(sample)
        for sample in samples
    )

    severity_text = (
        f"{mean_severity:.3f}" if mean_severity is not None else "--"
    )
    print(
        f"  Done. Mean severity: {severity_text} "
        f"(scored {len(severities)}/{len(samples)})"
    )
    print(f"  Remaining scorable samples without severity: {remaining}")
    print(f"  Saved in place: {file_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        help="Process JSON files immediately inside this directory.",
    )
    source.add_argument(
        "--recursive-dir",
        help="Process JSON files recursively beneath this directory.",
    )
    source.add_argument(
        "--files",
        nargs="+",
        help="Process these specific JSON files.",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help="Backfill empty hypotheses and summaries without calling Ollama.",
    )
    parser.add_argument(
        "--mirror-to-main",
        action="store_true",
        help="Also copy each updated file to writeup_results/benchmarks/main.",
    )
    args = parser.parse_args()

    if args.files:
        file_paths = [Path(value) for value in args.files]
    elif args.recursive_dir:
        file_paths = sorted(Path(args.recursive_dir).rglob("*.json"))
    else:
        file_paths = sorted(Path(args.input_dir).glob("*.json"))

    if not file_paths:
        raise SystemExit("No files found to process.")

    missing = [path for path in file_paths if not path.exists()]
    if missing:
        joined = "\n".join(f"  {path}" for path in missing)
        raise SystemExit(f"Input files not found:\n{joined}")

    print(f"Found {len(file_paths)} file(s) to process.")
    client = None if args.empty_only else Client(host=OLLAMA_HOST)

    for file_path in file_paths:
        process_file(
            file_path,
            client,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            empty_only=args.empty_only,
            mirror_to_main=args.mirror_to_main,
        )


if __name__ == "__main__":
    main()
