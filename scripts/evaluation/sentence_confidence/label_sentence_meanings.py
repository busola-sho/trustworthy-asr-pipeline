"""
scripts/evaluation/sentence_confidence/label_sentence_meanings.py

Labels each sentence in a selector output file with:
  - mar_verdict (True/False) — did meaning change?
  - severity (0-4) — how severe was the change?

These are the ground truth labels shared across all confidence methods.

Output format:
{
  "dataset": str,
  "combo_file": str,
  "rows": [
    {
      "dataset_index": int,
      "sent_pos": int,
      "hyp_sentence": str,
      "mar_verdict": bool,
      "severity": int,
    }
  ]
}

Usage:
    python scripts/evaluation/sentence_confidence/label_sentence_meanings.py \
        --combo results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_commonvoice_qwen_sub150.json \
        --output results/sentence_confidence/sentence_labels_commonvoice.json

    python scripts/evaluation/sentence_confidence/label_sentence_meanings.py \
        --combo results/combinations_v2judge/context_v2_whisperx_confidence_sentences/context_v2_edacc_qwen_sub150.json \
        --output results/sentence_confidence/sentence_labels_edacc.json
"""

import json
import os
import argparse
import time
from ollama import Client
from src.judge import ollama_sentence_mar, ollama_sentence_severity

OLLAMA_HOST = "http://localhost:11434"


def run(combo_path: str, output_path: str, rerun: bool = False):
    if not os.path.exists(combo_path):
        print(f"ERROR: combo file not found: {combo_path}")
        return

    with open(combo_path) as f:
        combo = json.load(f)

    dataset  = combo.get("dataset", os.path.basename(combo_path))
    samples  = combo.get("samples", [])

    # resume if output exists
    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        rows = existing.get("rows", [])
        done = {(r["dataset_index"], r["sent_pos"]) for r in rows}
        print(f"Resuming — {len(rows)} rows already labelled")
    else:
        rows = []
        done = set()

    client     = Client(host=OLLAMA_HOST)
    start_time = time.time()
    new_rows   = 0

    for s in samples:
        if s.get("skipped") or s.get("error"):
            continue

        ref           = s.get("ref", "")
        sent_confs    = s.get("sentence_confidences", [])
        dataset_index = s.get("dataset_index")

        if not ref or not sent_confs or dataset_index is None:
            continue

        for sent_pos, sc in enumerate(sent_confs):
            key = (dataset_index, sent_pos)
            if key in done:
                continue

            hyp_sentence = sc.get("sentence", "").strip()
            if not hyp_sentence:
                continue

            mar_verdict = ollama_sentence_mar(client, ref, hyp_sentence)
            severity    = ollama_sentence_severity(client, ref, hyp_sentence)

            rows.append({
                "dataset_index": dataset_index,
                "sent_pos":      sent_pos,
                "hyp_sentence":  hyp_sentence,
                "mar_verdict":   mar_verdict,
                "severity":      severity,
            })
            done.add(key)
            new_rows += 1

            if new_rows % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  {len(rows)} rows total, {new_rows} new ({elapsed:.0f}s)")
                _save(output_path, dataset, combo_path, rows)

    _save(output_path, dataset, combo_path, rows)

    # summary
    valid = [r for r in rows
             if r.get("mar_verdict") is not None
             and r.get("severity")   is not None]
    n         = len(valid)
    n_errors  = sum(1 for r in valid if r["mar_verdict"])
    mean_sev  = sum(r["severity"] for r in valid) / n if n else 0

    print(f"\nDone.")
    print(f"  Total labelled: {n}")
    print(f"  MAR rate:       {n_errors/n*100:.1f}%")
    print(f"  Mean severity:  {mean_sev:.3f}")
    print(f"  Saved: {output_path}")


def _save(output_path, dataset, combo_path, rows):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "dataset":    dataset,
            "combo_file": combo_path,
            "rows":       rows,
        }, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Label sentences with MAR verdict and severity score"
    )
    parser.add_argument("--combo",  required=True,
                        help="Path to selector output JSON")
    parser.add_argument("--output", required=True,
                        help="Path to save sentence labels JSON")
    parser.add_argument("--rerun",  action="store_true",
                        help="Rerun from scratch")
    args = parser.parse_args()
    run(args.combo, args.output, rerun=args.rerun)


if __name__ == "__main__":
    main()