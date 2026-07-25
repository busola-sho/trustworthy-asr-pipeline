"""
add_summary_stats.py

Computes and writes top-level summary stats (mean_severity,
severity_distribution, judge label, scored/total counts) into a
benchmark JSON file, based on whatever per-sample "severity" values are
CURRENTLY in the file.

Needed because transfer_severity.py only ever copies per-sample severity
fields - it never touches or recomputes the top-level summary block, so
a merged file ends up with real per-sample severity but no aggregate
stats. Recomputing fresh (rather than copying the old file's summary
directly) is safer, since the merged sample set can differ slightly
after severity transfer/re-judging.

Usage:
    python add_summary_stats.py writeup_results/benchmarks/main/parakeet_commonvoice_merged.json
    python add_summary_stats.py writeup_results/benchmarks/main/*.json   (shell will expand this)
"""

import json
import sys

JUDGE_LABEL = "phi4:14b (direct prompt)"   # the locked severity judge used throughout


def add_summary(path: str):
    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", data) if isinstance(data, dict) else data

    valid = [s for s in samples if s.get("severity") is not None]
    severities = [s["severity"] for s in valid]

    if not severities:
        print(f"  {path}: no scored samples yet (0 with severity) - skipping summary")
        return

    mean_severity = round(sum(severities) / len(severities), 3)
    severity_distribution = {str(i): severities.count(i) for i in range(5)}

    summary_fields = {
        "judge": JUDGE_LABEL,
        "mean_severity": mean_severity,
        "severity_distribution": severity_distribution,
        "num_scored": len(valid),
        "num_samples": len(samples),
    }

    if isinstance(data, dict):
        data.update(summary_fields)
        output = data
    else:
        # samples-only file (list at top level) - wrap with the summary fields
        output = {**summary_fields, "samples": samples}

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  {path}: mean_severity={mean_severity}  "
          f"distribution={severity_distribution}  scored={len(valid)}/{len(samples)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python add_summary_stats.py <file1.json> [file2.json ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        add_summary(path)


if __name__ == "__main__":
    main()