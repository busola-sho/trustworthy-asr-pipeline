"""
rerunning/sentence_confidence/label_sentence_meanings.py

Labels each sentence in a selector output file with mar_verdict and
severity (ground truth for Method 4). Processes in batches (default
50 sentences), concurrent within each batch, saving progress after
every batch - not one giant all-at-once run with a single save at the
end, which gives no visibility into progress for large datasets.

Usage:
    python rerunning/sentence_confidence/label_sentence_meanings.py \
        --combo writeup_results/ensembles/naive_confscore/naive_confscore_english_dialects_gemma4sel_dev.json \
        --output results/sentence_confidence/sentence_labels_english_dialects_confscore.json
"""

import json
import os
import argparse
import time
from ollama import Client
from src.judge import ollama_sentence_mar, ollama_sentence_severity
from src.concurrent_ollama import run_concurrent

OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "phi4:14b"
MAX_WORKERS = int(os.environ.get("ENSEMBLE_MAX_WORKERS", "8"))
BATCH_SIZE  = 50


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run(combo_path, output_path, rerun=False, max_workers=MAX_WORKERS, batch_size=BATCH_SIZE):
    if not os.path.exists(combo_path):
        print(f"ERROR: combo file not found: {combo_path}")
        return

    with open(combo_path) as f:
        combo = json.load(f)

    dataset = combo.get("dataset", os.path.basename(combo_path))
    samples = combo.get("samples", [])

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        rows = existing.get("rows", [])
        done = {(r["dataset_index"], r["sent_pos"]) for r in rows}
        print(f"Resuming - {len(rows)} rows already labelled")
    else:
        rows, done = [], set()

    client = Client(host=OLLAMA_HOST)

    work_items = []
    for s in samples:
        if s.get("skipped") or s.get("error"):
            continue
        ref = s.get("ref", "")
        dataset_index = s.get("dataset_index")
        sent_confs = s.get("sentence_confidences", [])
        if not ref or not sent_confs or dataset_index is None:
            continue
        for sent_pos, sc in enumerate(sent_confs):
            key = (dataset_index, sent_pos)
            if key in done:
                continue
            hyp_sentence = sc.get("sentence", "").strip()
            if not hyp_sentence:
                continue
            work_items.append((dataset_index, sent_pos, ref, hyp_sentence,
                                sc.get("confidence"), sc.get("score")))

    print(f"{len(work_items)} sentences queued ({batch_size} per batch, {max_workers} workers per batch)")

    def _worker(item):
        _, _, ref, hyp, _, _ = item
        mar = ollama_sentence_mar(client, ref, hyp, model=JUDGE_MODEL)
        sev = ollama_sentence_severity(client, ref, hyp, model=JUDGE_MODEL)
        return mar, sev

    def save_progress():
        with open(output_path, "w") as f:
            json.dump({"dataset": dataset, "combo_file": combo_path, "judge": JUDGE_MODEL, "rows": rows},
                       f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    start_time = time.time()
    n_done = 0

    for batch_num, batch in enumerate(_chunked(work_items, batch_size), start=1):
        results = run_concurrent(batch, _worker, max_workers=max_workers, progress_every=0)
        for item, (mar, sev) in zip(batch, results):
            dataset_index, sent_pos, _, hyp_sentence, verbalized_conf, verbalized_score = item
            rows.append({
                "dataset_index":    dataset_index,
                "sent_pos":         sent_pos,
                "hyp_sentence":     hyp_sentence,
                "verbalized_conf":  verbalized_conf,
                "verbalized_score": verbalized_score,
                "mar_verdict":      mar,
                "severity":         sev,
            })
        n_done += len(batch)
        elapsed = time.time() - start_time
        print(f"  batch {batch_num}: {n_done}/{len(work_items)} labelled ({elapsed:.0f}s elapsed)")
        save_progress()   # crash-safe, and gives you something to check mid-run

    valid = [r for r in rows if r.get("mar_verdict") is not None and r.get("severity") is not None]
    n = len(valid)
    n_errors = sum(1 for r in valid if r["mar_verdict"])
    mean_sev = sum(r["severity"] for r in valid) / n if n else 0
    print(f"\nDone. Judge: {JUDGE_MODEL}")
    print(f"  Total labelled: {n}")
    print(f"  MAR rate:       {n_errors/n*100:.1f}%" if n else "  MAR rate: -")
    print(f"  Mean severity:  {mean_sev:.3f}")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rerun",  action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    run(args.combo, args.output, rerun=args.rerun, max_workers=args.max_workers, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
