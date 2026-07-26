"""
rerunning/add_severity_to_existing.py

Adds missing Phi-4 severity scores to existing benchmark JSON files.
Does not rerun ASR inference.

Important behaviour:
- samples with an empty hypothesis are NOT skipped;
- empty hypotheses are sent to the judge as "[EMPTY TRANSCRIPT]";
- the original stored hypothesis remains unchanged;
- samples with unusable references or explicitly skipped samples remain skipped;
- failed or unparsable judge outputs are recorded.

Usage:
    python rerunning/add_severity_to_existing.py --files file1.json file2.json

    python rerunning/add_severity_to_existing.py \
        --input-dir writeup_results/benchmarks/main
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from ollama import Client
from severity_judge_prompts import DIRECT_SEVERITY_PROMPT


OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"
NEW_OUTPUT_DIR = "writeup_results/benchmarks/main"
EMPTY_TRANSCRIPT_PLACEHOLDER = "[EMPTY TRANSCRIPT]"

TAG_ONLY_PATTERNS = [
    re.compile(r"^\s*<[^<>]+>\s*$"),
    re.compile(r"^\s*\[[^\[\]]+\]\s*$"),
    re.compile(r"^\s*\([^()]+\)\s*$"),
]


def parse_severity(text: Optional[str]) -> Optional[int]:
    if not text:
        return None

    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    stripped = text.strip()
    if re.fullmatch(r"[0-4]", stripped):
        return int(stripped)

    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def is_tag_only_reference(text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return any(pattern.fullmatch(text.strip()) for pattern in TAG_ONLY_PATTERNS)


def call_judge(client: Client, prompt: str) -> str:
    response = client.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_predict": 80,
        },
        think=False,
    )
    return response.message.content or ""


def ollama_severity(
    client: Client,
    ref: str,
    hyp: str,
    retries: int = 2,
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """
    Returns:
        severity, raw_response, failure_reason
    """
    original_prompt = DIRECT_SEVERITY_PROMPT.format(
        reference=ref,
        hypothesis=hyp,
    )

    last_response = None
    last_error = None

    for attempt in range(retries + 1):
        try:
            prompt = original_prompt
            if attempt > 0:
                prompt += (
                    "\n\nReturn exactly one line in this format:\n"
                    "Severity: X\n"
                    "where X is one integer from 0 to 4. Do not add other text."
                )

            last_response = call_judge(client, prompt)
            severity = parse_severity(last_response)

            if severity is not None:
                return severity, last_response, None

            last_error = "unparsable_judge_response"

        except Exception as exc:
            last_error = f"judge_exception: {exc}"

        if attempt < retries:
            time.sleep(1.0)

    return None, last_response, last_error


def process_file(file_path: Path, client: Client) -> None:
    print(f"\n── {file_path.name} ──")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        print("  No samples found, skipping.")
        return

    new_output_path = Path(NEW_OUTPUT_DIR) / file_path.name
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

    n_total = len(samples)
    n_already_done = sum(
        1 for sample in samples
        if sample.get("severity") is not None
    )

    print(
        f"  {n_total} samples, "
        f"{n_already_done} already have severity, resuming..."
    )

    def save_progress() -> None:
        with new_output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if file_path.resolve() != new_output_path.resolve():
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    start_time = time.time()
    n_scored_this_run = 0
    n_empty_hyp_scored = 0
    n_intentionally_skipped = 0
    n_failed = 0

    for i, sample in enumerate(samples):
        if sample.get("severity") is not None:
            continue

        ref = sample.get("ref")
        hyp = sample.get("hyp")

        skip_reason = None

        if sample.get("skipped"):
            skip_reason = "sample_marked_skipped"
        elif not isinstance(ref, str) or not ref.strip():
            skip_reason = "missing_or_empty_reference"
        elif is_tag_only_reference(ref):
            skip_reason = "tag_only_reference"
        elif hyp is None:
            skip_reason = "missing_hypothesis_field"
        elif not isinstance(hyp, str):
            hyp = str(hyp)

        if skip_reason:
            sample["severity"] = None
            sample["severity_status"] = "skipped"
            sample["severity_skip_reason"] = skip_reason
            n_intentionally_skipped += 1

            print(
                f"  [{i + 1}/{n_total}] SKIP: {skip_reason} "
                f"(sample_index={sample.get('sample_index', i)})"
            )
            continue

        judge_hyp = hyp.strip()
        used_empty_placeholder = False

        if not judge_hyp:
            judge_hyp = EMPTY_TRANSCRIPT_PLACEHOLDER
            used_empty_placeholder = True
            sample["severity_input_hyp"] = EMPTY_TRANSCRIPT_PLACEHOLDER

        severity, raw_response, failure_reason = ollama_severity(
            client,
            ref.strip(),
            judge_hyp,
        )

        sample["severity"] = severity
        sample["judge_raw_response"] = raw_response

        if severity is None:
            sample["severity_status"] = "failed"
            sample["severity_failure_reason"] = failure_reason
            n_failed += 1

            print(
                f"  [{i + 1}/{n_total}] FAILED: {failure_reason} "
                f"(sample_index={sample.get('sample_index', i)})"
            )
        else:
            sample["severity_status"] = "scored"
            sample.pop("severity_failure_reason", None)
            sample.pop("severity_skip_reason", None)
            n_scored_this_run += 1

            if used_empty_placeholder:
                n_empty_hyp_scored += 1

            suffix = " [empty hypothesis]" if used_empty_placeholder else ""
            print(
                f"  [{i + 1}/{n_total}] severity={severity}{suffix} "
                f"({time.time() - start_time:.0f}s elapsed)"
            )

        if (n_scored_this_run + n_failed) % 10 == 0:
            save_progress()

    severities = [
        sample["severity"]
        for sample in samples
        if sample.get("severity") is not None
    ]

    mean_severity = (
        sum(severities) / len(severities)
        if severities
        else None
    )

    data["judge"] = f"{JUDGE_MODEL} (direct prompt)"
    data["mean_severity"] = (
        round(mean_severity, 3)
        if mean_severity is not None
        else None
    )
    data["severity_distribution"] = {
        str(level): severities.count(level)
        for level in range(5)
    }
    data["severity_coverage"] = {
        "total_samples": n_total,
        "scored": len(severities),
        "unscored": n_total - len(severities),
        "scored_this_run": n_scored_this_run,
        "empty_hypotheses_scored_this_run": n_empty_hyp_scored,
        "intentionally_skipped_this_run": n_intentionally_skipped,
        "judge_failures_this_run": n_failed,
    }

    save_progress()

    sev_str = (
        f"{mean_severity:.3f}"
        if mean_severity is not None
        else "—"
    )

    print(
        f"  Done. Mean severity: {sev_str} "
        f"(scored {len(severities)}/{n_total})"
    )
    print(
        f"  This run: +{n_scored_this_run} scored "
        f"({n_empty_hyp_scored} empty hypotheses), "
        f"{n_intentionally_skipped} skipped, "
        f"{n_failed} judge failures"
    )
    print(f"  Saved: {new_output_path}")

    if file_path.resolve() != new_output_path.resolve():
        print(f"  Saved: {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing JSON result files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific JSON files to process.",
    )
    args = parser.parse_args()

    if not args.input_dir and not args.files:
        raise SystemExit("Provide either --input-dir or --files")

    if args.files:
        file_paths = [Path(path) for path in args.files]
    else:
        file_paths = sorted(Path(args.input_dir).glob("*.json"))

    missing_files = [path for path in file_paths if not path.exists()]
    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files)
        raise SystemExit(f"These files do not exist:\n{missing_text}")

    if not file_paths:
        raise SystemExit("No files found to process.")

    print(f"Found {len(file_paths)} file(s) to process.")

    client = Client(host=OLLAMA_HOST)

    for file_path in file_paths:
        process_file(file_path, client)


if __name__ == "__main__":
    main()