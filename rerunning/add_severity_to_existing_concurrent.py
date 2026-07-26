"""
rerunning/add_severity_to_existing_concurrent.py

Adds severity scores (Phi-4 + direct prompt, the locked judge) to EXISTING
benchmark JSON files - does NOT re-run any ASR inference. Reads the ref/hyp
pairs already present in each file and judges them.

CONCURRENT VERSION: fires judge calls through a thread pool instead of
one at a time. See src/concurrent_ollama.py's docstring for the required
OLLAMA_NUM_PARALLEL server-side setting - client-side concurrency alone
does nothing without it.

BATCHED FOR CRASH-SAFETY: work is processed in chunks (default 50
samples), each chunk run concurrently, with a save after every chunk
completes - not one giant all-at-once thread pool. Severity files can be
large (English Dialects has 1700+ samples), so saving only at the very
end would mean a job timeout or crash loses ALL progress on that file,
not just the in-flight chunk. This preserves the original script's
resumability, just at chunk granularity instead of per-sample.

Writes to BOTH locations, same filename:
  - writeup_results/benchmarks/main/{filename}  (new)
  - the file's original location, updated in place  (old, kept for continuity)

Resumable: skips any sample that already has a non-null "severity" field.

Usage:
    python rerunning/add_severity_to_existing_concurrent.py --input-dir results/benchmarks/main
    python rerunning/add_severity_to_existing_concurrent.py --files writeup_results/ensembles/naive_confidence/naive_conf_english_dialects_gemma4_p20_dev.json
"""

import json
import os
import re
import argparse
import time
from pathlib import Path

from severity_judge_prompts import DIRECT_SEVERITY_PROMPT
from ollama import Client
from src.concurrent_ollama import run_concurrent

OLLAMA_HOST  = "http://localhost:11434"
JUDGE_MODEL  = "phi4:14b"   # locked severity judge (Phi-4 + direct, QWK=0.783)

NEW_OUTPUT_DIR = "writeup_results/benchmarks/main"
MAX_WORKERS = 1     # tune to roughly match OLLAMA_NUM_PARALLEL on the server
BATCH_SIZE = 50     # save progress after every N samples judged, not just at the end


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


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def process_file(file_path: Path, client: Client, max_workers: int = MAX_WORKERS,
                  batch_size: int = BATCH_SIZE):
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

    # ── PHASE A: identify which samples actually need judging ──
    # (already-scored samples untouched; unscoreable ones get severity=None
    # immediately, same behaviour as the original script)
    work_items = []   # each: (sample_list_index, ref, hyp)
    for i, sample in enumerate(samples):
        if sample.get("severity") is not None:
            continue
        ref = sample.get("ref")
        hyp = sample.get("hyp")
        if not ref or not hyp or sample.get("skipped"):
            sample["severity"] = None
            continue
        work_items.append((i, ref, hyp))

    print(f"  {len(work_items)} samples queued for concurrent severity judging "
          f"(batches of {batch_size}, {max_workers} workers per batch)")

    start_time = time.time()
    n_judged = 0

    def _worker(item):
        _, ref, hyp = item
        return ollama_severity(client, ref, hyp)

    # ── PHASE B: judge in batches, concurrent within each batch, save after each ──
    for batch_num, batch in enumerate(_chunked(work_items, batch_size), start=1):
        results = run_concurrent(batch, _worker, max_workers=max_workers, progress_every=0)

        for (idx, ref, hyp), result in zip(batch, results):
            severity, raw_response = result if result is not None else (None, None)
            samples[idx]["severity"] = severity
            samples[idx]["judge_raw_response"] = raw_response

        n_judged += len(batch)
        elapsed = time.time() - start_time
        print(f"  batch {batch_num}: {n_judged}/{len(work_items)} judged "
              f"({elapsed:.0f}s elapsed this run)")

        save_progress()   # crash-safe: save after every batch, not just at the very end

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
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help="Concurrent Ollama calls per batch (tune to match "
                             "OLLAMA_NUM_PARALLEL on the server)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Samples judged per batch before saving progress")
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
        process_file(file_path, client, max_workers=args.max_workers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()