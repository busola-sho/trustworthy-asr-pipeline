"""
rerunning/add_severity_to_existing.py

Adds severity scores (Phi-4 + direct prompt, the locked judge) to EXISTING
benchmark JSON files - does NOT re-run any ASR inference. Reads the ref/hyp
pairs already present in each file and judges them.

Writes to BOTH locations, same filename:
  - writeup_results/benchmarks/main/{filename}  (new)
  - the file's original location, updated in place  (old, kept for continuity)

Resumable: skips any sample that already has a non-null "severity" field.

Usage:
    # process every *.json file in a directory
    python rerunning/add_severity_to_existing.py --input-dir results/benchmarks/main

    # process specific files only
    python rerunning/add_severity_to_existing.py --files results/benchmarks/main/whisper_commonvoice_20260524_083930.json results/benchmarks/main/qwen_edacc_20260525_204314.json
"""

import json
import os
import re
import argparse
import time
from pathlib import Path

from severity_judge_prompts import DIRECT_SEVERITY_PROMPT
from ollama import Client

OLLAMA_HOST  = "http://localhost:11434"
JUDGE_MODEL  = "phi4:14b"   # locked severity judge (Phi-4 + direct, QWK=0.783)

NEW_OUTPUT_DIR = "writeup_results/benchmarks/main"


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
        except Exception as e:
            if attempt == retries:
                print(f"    ERROR (severity judge): {e}")
                return None, None
            time.sleep(1.0)
    return None, text


def process_file(file_path: Path, client: Client):
    print(f"\n── {file_path.name} ──")
    with open(file_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if not samples:
        print("  No samples found, skipping.")
        return

    new_output_path = Path(NEW_OUTPUT_DIR) / file_path.name
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

    n_total = len(samples)
    n_already_done = sum(1 for s in samples if s.get("severity") is not None)
    print(f"  {n_total} samples, {n_already_done} already have severity, resuming...")

    def save_progress():
        with open(new_output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    start_time = time.time()
    n_processed_this_run = 0

    for i, sample in enumerate(samples):
        # skip already-scored samples (resume support)
        if sample.get("severity") is not None:
            continue

        # skip samples with no scoreable ref/hyp (e.g. previously flagged
        # as skipped due to tag-only reference or ignore-time-segment)
        ref = sample.get("ref")
        hyp = sample.get("hyp")
        if not ref or not hyp or sample.get("skipped"):
            sample["severity"] = None
            continue

        severity, raw_response = ollama_severity(client, ref, hyp)
        sample["severity"] = severity
        sample["judge_raw_response"] = raw_response
        n_processed_this_run += 1

        print(f"  [{i+1}/{n_total}] severity={severity}  "
              f"({time.time()-start_time:.0f}s elapsed this run)")

        if n_processed_this_run % 10 == 0:
            save_progress()

    # recompute summary stats
    severities = [s["severity"] for s in samples if s.get("severity") is not None]
    mean_severity = sum(severities) / len(severities) if severities else None
    severity_distribution = {str(i): severities.count(i) for i in range(5)}

    data["judge"] = f"{JUDGE_MODEL} (direct prompt)"
    data["mean_severity"] = round(mean_severity, 3) if mean_severity is not None else None
    data["severity_distribution"] = severity_distribution

    save_progress()

    sev_str = f"{mean_severity:.3f}" if mean_severity is not None else "—"
    print(f"  Done. Mean severity: {sev_str}  (scored {len(severities)}/{n_total})")
    print(f"  Saved: {new_output_path}")
    print(f"  Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None,
                        help="Directory - processes every *.json file inside")
    parser.add_argument("--files", nargs="+", default=None,
                        help="Specific file paths to process instead of a directory")
    args = parser.parse_args()

    if not args.input_dir and not args.files:
        raise SystemExit("Provide either --input-dir or --files")

    if args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        file_paths = sorted(Path(args.input_dir).glob("*.json"))

    if not file_paths:
        raise SystemExit("No files found to process.")

    print(f"Found {len(file_paths)} file(s) to process.")

    client = Client(host=OLLAMA_HOST)
    for file_path in file_paths:
        process_file(file_path, client)


if __name__ == "__main__":
    main()