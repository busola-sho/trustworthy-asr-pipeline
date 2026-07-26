"""
build_leaderboard.py

Scans your ensemble result JSON files and builds:
  1. One table PER DATASET (commonvoice, english_dialects, edacc), rows =
     techniques, sorted by mean_severity (primary quality metric) then WER.
  2. ONE combined table: rows = techniques, columns = each dataset's WER
     and mean_severity, plus an AVERAGE column across all 3 datasets.

Relies on fields every ensemble script already writes into its output
JSON ("approach", "dataset", "split", "corpus_wer", "mean_severity",
"num_scored"/"num_samples") rather than parsing filenames.

DEDUPES by (approach, dataset, percentile): if the same technique+dataset
combo is found in more than one file (e.g. a duplicate copy sitting in a
folder like "ensembles_again"), only ONE entry is kept, and a warning is
printed listing the duplicate paths so you can clean them up if you want.

Prints proper fixed-width, aligned plain-text tables (readable directly
in a terminal) - NOT raw markdown pipe syntax, which only renders
correctly in a markdown viewer. Pass --markdown-out to ALSO write a
markdown version to a file, useful for pasting into the dissertation
itself later.

Usage:
    python build_leaderboard.py
    python build_leaderboard.py --split dev
    python build_leaderboard.py --markdown-out leaderboard.md
    python build_leaderboard.py --roots writeup_results/ensembles writeup_results/voting/rover writeup_results/benchmarks/main
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from jiwer import wer as compute_wer
from src.text_normalise import normalise
from src.splits import get_indices_for_split

DEFAULT_ROOTS = [
    "writeup_results/ensembles",
    "writeup_results/voting",
    "writeup_results/benchmarks/main",
]

DATASET_ORDER = ["commonvoice", "english_dialects", "edacc"]

# some older benchmark files predate the project's naming standardization
# and use a different spelling for the same dataset - same normalization
# already applied in selector_ablation.py for candidate_pool.json
DATASET_NAME_ALIASES = {
    "common_voice": "commonvoice",
    "edinburgh_international_accents": "edacc",
    "english_dialects_scots": "english_dialects",
}


def normalize_dataset_name(raw_name: str) -> str:
    return DATASET_NAME_ALIASES.get(raw_name, raw_name)


def find_result_files(roots):
    results = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or "dataset" not in data:
                continue
            # ensemble technique files have "approach"; individual ASR
            # model benchmark files have "model" instead - both are
            # useful, tagged differently so the combined table can single
            # out just the best baseline model rather than showing all 4
            if "approach" in data or "model" in data:
                results.append((path, data))
    return results


MODEL_NAME_ALIASES = {
    "wav2vec2": "wav2vec2",
    "parakeet": "parakeet",
    "qwen3-asr": "qwen",
    "qwen3asr": "qwen",   # some files (e.g. Shetland's) use this form with no hyphen
    "whisperx": "whisperx",
}


def normalize_model_name(raw_name: str) -> str:
    """Normalise benchmark model names to stable short labels.

    Some result files store full checkpoint paths while others use short names.
    We also strip punctuation so forms such as ``qwen3-asr`` and ``qwen3asr``
    are grouped together rather than appearing as separate techniques.
    """
    lowered = raw_name.lower()
    compact = lowered.replace("-", "").replace("_", "").replace("/", "")

    if "qwen3asr" in compact:
        return "qwen"
    if "wav2vec2" in compact:
        return "wav2vec2"
    if "parakeet" in compact:
        return "parakeet"
    if "whisper" in compact:
        return "whisperx"

    return raw_name


def extract_row(path, data):
    is_baseline = "approach" not in data and "model" in data
    approach = data.get("approach") or data.get("model", "unknown")
    if is_baseline:
        approach = normalize_model_name(approach)
    dataset = normalize_dataset_name(data.get("dataset", "unknown"))
    split = data.get("split", data.get("full_dataset", None))
    corpus_wer = data.get("corpus_wer")
    mean_severity = data.get("mean_severity")
    num_scored = data.get("num_scored", data.get("num_samples"))
    percentile = data.get("percentile")

    return {
        "path": str(path),
        "approach": approach,
        "dataset": dataset,
        "split": split,
        "corpus_wer": corpus_wer,
        "mean_severity": mean_severity,
        "num_scored": num_scored,
        "percentile": percentile,
        "is_baseline": is_baseline,
        "raw_data": data,   # kept so baseline rows can be recomputed restricted to a split
    }


def recompute_baseline_for_split(row, split):
    """
    Baseline (individual ASR model) files are full-dataset runs with no
    dev/test concept of their own - their top-level corpus_wer/mean_severity
    cover EVERY sample (dev + test + calibration all mixed together), which
    isn't a fair comparison against an ensemble technique that was only
    ever run on the "dev" split. This recomputes WER/severity restricted to
    exactly the same sample indices get_indices_for_split() gives the
    ensemble scripts, so the comparison is apples-to-apples.

    Returns a NEW row dict with corpus_wer/mean_severity/num_scored
    replaced by the split-restricted values. Falls back to the original
    (full-dataset) values with a warning if samples/severity data isn't
    available to recompute from.
    """
    data = row["raw_data"]
    samples = data.get("samples")
    dataset = row["dataset"]

    # Shetland is the untouched external held-out set - it's NEVER divided
    # into dev/test, always used in its entirety exactly once. Silently
    # applying the normal dev/70-test/30 split logic to it would restrict
    # the "recomputed" severity to only ~70 of its 100 samples (whatever
    # random subset the dev/test split machinery happens to assign),
    # which is wrong for this dataset specifically - force "full" here
    # regardless of what --split was requested for the other datasets.
    if dataset == "shetland" and split != "full":
        print(f"  NOTE: {row['path']} is Shetland - using split='full' (all samples) "
              f"instead of '{split}', since Shetland is never divided into dev/test")
        split = "full"

    if not samples:
        print(f"  WARNING: {row['path']} has no per-sample data to restrict to '{split}' - "
              f"using its full-dataset stats as-is (not split-matched)")
        return row

    # backfill sample_index from list position for older files that predate
    # per-sample indexing - same convention used throughout the rest of the
    # pipeline (verified safe via compare_index_alignment.sh earlier)
    for i, s in enumerate(samples):
        if s.get("sample_index") is None:
            s["sample_index"] = i

    try:
        split_indices = set(get_indices_for_split(dataset, split))
    except (ValueError, KeyError) as e:
        print(f"  WARNING: could not get '{split}' indices for {dataset} ({e}) - "
              f"using full-dataset stats for {row['path']}")
        return row

    subset = [s for s in samples if s.get("sample_index") in split_indices]

    refs, hyps = [], []
    severities = []
    for s in subset:
        if s.get("skipped") or s.get("error"):
            continue
        ref, hyp = s.get("ref"), s.get("hyp")
        if ref and hyp:
            refs.append(normalise(ref))
            hyps.append(normalise(hyp))
        if s.get("severity") is not None:
            severities.append(s["severity"])

    new_wer = compute_wer(refs, hyps) if refs else None
    new_severity = sum(severities) / len(severities) if severities else None

    new_row = dict(row)
    if new_wer is None and new_severity is None:
        print(f"  WARNING: {row['path']} had samples for '{split}' but none had usable "
              f"ref/hyp/severity fields - keeping original full-dataset stats instead "
              f"of discarding them (not split-matched, flagged here so you know)")
        return row
    # keep whichever of WER/severity recomputed successfully; fall back to the
    # original full-dataset value for whichever one didn't (rather than losing
    # a perfectly good existing number just because e.g. per-sample severity
    # happened to be missing while per-sample ref/hyp were present)
    new_row["corpus_wer"] = new_wer if new_wer is not None else row["corpus_wer"]
    new_row["mean_severity"] = new_severity if new_severity is not None else row["mean_severity"]
    new_row["num_scored"] = len(refs) if refs else row["num_scored"]

    # CRITICAL: also actually trim raw_data's "samples" list to just the
    # split-restricted subset - without this, the summary fields above are
    # correctly restricted, but anything downstream that reads
    # row["raw_data"]["samples"] directly (e.g. compute_pooled_metrics for
    # the pooled/micro-average table) would silently read the ENTIRE
    # unfiltered file instead, defeating the whole point of this function
    new_row["raw_data"] = {**row["raw_data"], "samples": subset}
    return new_row


def compute_pooled_metrics(rows_for_one_technique):
    """
    Computes POOLED (micro-average) WER and severity across all of a
    technique's datasets combined - i.e. treating every sample from
    every dataset as one single pool, rather than averaging each
    dataset's own mean (which is what the "macro" combined tables show).

    This matters because averaging per-dataset WER percentages is
    mathematically wrong for a pooled number - WER's numerator/denominator
    relationship isn't linear across datasets with different total
    reference-word counts. The correct pooled WER is:
        sum(S+D+I edit operations across every sample in every dataset)
        -------------------------------------------------------------
        sum(reference word count across every sample in every dataset)
    jiwer.wer() already computes exactly this when given a single
    combined list of refs/hyps (not per-dataset lists averaged
    afterward) - so pooling is just "concatenate every dataset's
    ref/hyp pairs, call wer() once", not a manual S+D+I sum.

    Pooled severity is simply the mean of every individual sample's
    severity value across every dataset, pooled together.

    rows_for_one_technique: list of rows (one per dataset) for the SAME
    approach/technique - each row's raw_data["samples"] is read directly,
    same skip logic as recompute_baseline_for_split.

    Returns (pooled_wer, pooled_severity, total_n) - any of the first two
    can be None if no usable data was found across all provided rows.
    """
    all_refs, all_hyps = [], []
    all_severities = []

    for row in rows_for_one_technique:
        samples = row["raw_data"].get("samples")
        if not samples:
            continue
        for s in samples:
            if s.get("skipped") or s.get("error"):
                continue
            ref, hyp = s.get("ref"), s.get("hyp")
            if ref and hyp:
                all_refs.append(normalise(ref))
                all_hyps.append(normalise(hyp))
            if s.get("severity") is not None:
                all_severities.append(s["severity"])

    pooled_wer = compute_wer(all_refs, all_hyps) if all_refs else None
    pooled_severity = sum(all_severities) / len(all_severities) if all_severities else None
    return pooled_wer, pooled_severity, len(all_severities)


def build_pooled_table_rows(rows, datasets=None):
    """Build pooled/micro-average rows.

    Parameters
    ----------
    rows:
        Result rows to pool.
    datasets:
        Optional iterable of dataset names to include. Passing
        ``DATASET_ORDER`` keeps Shetland out of the in-domain pooled table.

    This explicit filter prevents a method that happens to have a Shetland
    result from receiving 100 extra samples while the other methods do not.
    """
    if datasets is not None:
        allowed = set(datasets)
        rows = [r for r in rows if r["dataset"] in allowed]

    by_approach = defaultdict(list)
    for r in rows:
        by_approach[r["approach"]].append(r)

    results = []
    for approach, approach_rows in by_approach.items():
        pooled_wer, pooled_sev, n = compute_pooled_metrics(approach_rows)
        results.append((approach, pooled_sev, pooled_wer, n))

    results.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 999))

    return [
        [approach, fmt_sev(sev), fmt_wer(wer_val), n]
        for approach, sev, wer_val, n in results
    ]


def dedupe_rows(rows):
    """Keeps one entry per (approach, dataset, percentile) - warns about
    any duplicates found so you can clean up the duplicate files."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["approach"], r["dataset"], r["percentile"])
        groups[key].append(r)

    deduped = []
    dupes_found = []
    for key, group in groups.items():
        deduped.append(group[0])
        if len(group) > 1:
            dupes_found.append((key, [g["path"] for g in group]))

    if dupes_found:
        print(f"\nFound {len(dupes_found)} technique+dataset combo(s) with duplicate files "
              f"(kept the first, ignored the rest):")
        for (approach, dataset, pct), paths in dupes_found:
            label = f"{approach} (p{pct})" if pct is not None else approach
            print(f"  {label} / {dataset}:")
            for p in paths:
                print(f"    - {p}")

    return deduped


def fmt_wer(wer):
    return f"{wer*100:.2f}%" if wer is not None else "-"


def fmt_sev(sev):
    return f"{sev:.3f}" if sev is not None else "-"


def print_table(headers, rows, aligns=None):
    """Prints a proper fixed-width, aligned plain-text table - readable
    directly in a terminal, unlike raw markdown pipe syntax."""
    n_cols = len(headers)
    aligns = aligns or ["<"] * n_cols  # '<' left, '>' right

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def format_row(cells):
        parts = []
        for cell, width, align in zip(cells, col_widths, aligns):
            parts.append(f"{str(cell):{align}{width}}")
        return "  ".join(parts)

    header_line = format_row(headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(format_row(row))


def build_markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def technique_label(r):
    label = r["approach"]
    if r.get("percentile") is not None:
        label += f" (p{r['percentile']})"
    return label


def build_dataset_table_rows(dataset_rows):
    def sort_key(r):
        sev = r["mean_severity"]
        return (sev is None, sev if sev is not None else 999)

    table_rows = []
    for r in sorted(dataset_rows, key=sort_key):
        table_rows.append([
            technique_label(r),
            fmt_sev(r["mean_severity"]),
            fmt_wer(r["corpus_wer"]),
            r["num_scored"],
        ])
    return table_rows


def build_combined_table_rows(all_rows):
    by_approach = defaultdict(dict)
    for r in all_rows:
        by_approach[r["approach"]][r["dataset"]] = r

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def sort_key(approach):
        sevs = [by_approach[approach].get(ds, {}).get("mean_severity") for ds in DATASET_ORDER]
        avg_sev = avg(sevs)
        return (avg_sev is None, avg_sev if avg_sev is not None else 999)

    table_rows = []
    for approach in sorted(by_approach.keys(), key=sort_key):
        sevs, wers = [], []
        row = [approach]
        for ds in DATASET_ORDER:
            r = by_approach[approach].get(ds)
            sev = r["mean_severity"] if r else None
            wer = r["corpus_wer"] if r else None
            sevs.append(sev)
            wers.append(wer)
            row.append(fmt_sev(sev))
            row.append(fmt_wer(wer))
        row.append(fmt_sev(avg(sevs)))
        row.append(fmt_wer(avg(wers)))
        table_rows.append(row)
    return table_rows


def pick_top_n(rows, top_n, label="technique"):
    """Groups rows by 'approach' (works for both baseline model names and
    ensemble technique names), computes average severity across whichever
    datasets each has results for, and returns the rows for the TOP N
    best-averaging ones. Generic version of the old pick_best_baseline -
    reused for both baselines and ensemble techniques so the final
    top-2-vs-top-2 comparison table can be built the same way for both."""
    by_group = defaultdict(dict)
    for r in rows:
        by_group[r["approach"]][r["dataset"]] = r

    def avg_severity(name):
        sevs = [by_group[name].get(ds, {}).get("mean_severity") for ds in DATASET_ORDER]
        sevs = [s for s in sevs if s is not None]
        return sum(sevs) / len(sevs) if sevs else None

    candidates = [(m, avg_severity(m)) for m in by_group]
    candidates = [(m, s) for m, s in candidates if s is not None]
    if not candidates:
        return []

    candidates.sort(key=lambda x: x[1])
    top = candidates[:top_n]

    print(f"\nTop {len(top)} {label}(s) by avg severity (macro-average over "
          f"{', '.join(DATASET_ORDER)} only - Shetland is never folded into this "
          f"average, even if a technique has Shetland data too):")
    for rank, (name, avg_sev) in enumerate(top, start=1):
        # count only datasets actually included in avg_severity's calculation -
        # NOT every dataset by_group[name] happens to have (which could include
        # Shetland, silently overcounting relative to what was actually averaged)
        n_datasets = sum(1 for ds in DATASET_ORDER if by_group[name].get(ds, {}).get("mean_severity") is not None)
        print(f"  #{rank}: {name} (avg severity {avg_sev:.3f} across {n_datasets} dataset(s))")

    result_rows = []
    for name, _ in top:
        # The final in-domain comparison must use only the three datasets in
        # DATASET_ORDER. Shetland remains a separate external-test table.
        for ds in DATASET_ORDER:
            row = by_group[name].get(ds)
            if row is not None:
                result_rows.append(row)
    return result_rows


def best_baseline_per_dataset(baseline_rows):
    """Finds the single best baseline row for EACH dataset independently
    (lowest severity), rather than picking one 'overall best' model by
    average across datasets - since no single baseline model dominates
    every dataset, an average can be misleading about what actually wins
    where. Returns {dataset: best_row}."""
    by_dataset = defaultdict(list)
    for r in baseline_rows:
        by_dataset[r["dataset"]].append(r)

    best = {}
    for ds, rows in by_dataset.items():
        scored = [r for r in rows if r["mean_severity"] is not None]
        if scored:
            best[ds] = min(scored, key=lambda r: r["mean_severity"])
    return best


def best_ensemble_per_dataset(ensemble_rows):
    """Same idea as best_baseline_per_dataset, for ensemble techniques."""
    by_dataset = defaultdict(list)
    for r in ensemble_rows:
        by_dataset[r["dataset"]].append(r)

    best = {}
    for ds, rows in by_dataset.items():
        scored = [r for r in rows if r["mean_severity"] is not None]
        if scored:
            best[ds] = min(scored, key=lambda r: r["mean_severity"])
    return best


def build_head_to_head_rows(best_baselines, best_ensembles):
    rows = []
    for ds in DATASET_ORDER:
        b = best_baselines.get(ds)
        e = best_ensembles.get(ds)
        if not b or not e:
            continue
        winner = "ensemble" if e["mean_severity"] < b["mean_severity"] else "baseline"
        rows.append([
            ds,
            b["approach"], fmt_sev(b["mean_severity"]), fmt_wer(b["corpus_wer"]),
            e["approach"], fmt_sev(e["mean_severity"]), fmt_wer(e["corpus_wer"]),
            winner,
        ])
    return rows


def print_pool_diagnostics(rows, title):
    """Print the usable per-dataset sample counts that will enter pooling.

    This makes it immediately obvious if an old/full-dataset result file has
    slipped through split restriction.
    """
    print(f"\nPool diagnostics: {title}")
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] not in DATASET_ORDER:
            continue
        samples = row["raw_data"].get("samples") or []
        usable = [
            s for s in samples
            if not s.get("skipped")
            and not s.get("error")
            and s.get("severity") is not None
        ]
        grouped[row["approach"]].append((row["dataset"], len(usable), row["path"]))

    for approach in sorted(grouped):
        counts = ", ".join(f"{ds}={n}" for ds, n, _ in sorted(grouped[approach]))
        total = sum(n for _, n, _ in grouped[approach])
        print(f"  {approach}: {counts} -> total={total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--split", default="dev",
                        help="Only include results for this split (default: dev). "
                             "Pass 'all' to include every split found.")
    parser.add_argument("--top-baselines", type=int, default=2,
                        help="How many top individual ASR model baselines to include in the "
                             "final comparison table, ranked by average severity (default: 2)")
    parser.add_argument("--top-ensembles", type=int, default=2,
                        help="How many top ensemble techniques to include in the final "
                             "comparison table, ranked by average severity (default: 2)")
    parser.add_argument("--markdown-out", default=None,
                        help="Also write a markdown version of all tables to this file "
                             "(for pasting into the dissertation later)")
    args = parser.parse_args()

    found = find_result_files(args.roots)
    print(f"Scanned {args.roots} -> found {len(found)} result file(s) with approach+dataset fields")

    rows = [extract_row(path, data) for path, data in found]

    # split into ensemble vs baseline BEFORE applying the --split filter -
    # baseline (individual ASR model) files are full-dataset runs with no
    # dev/test split concept, so filtering by split would silently drop
    # them entirely (their "split" field is None, never equal to "dev")
    ensemble_candidates = [r for r in rows if not r["is_baseline"]]
    baseline_candidates = [r for r in rows if r["is_baseline"]]
    print(f"  {len(ensemble_candidates)} ensemble-technique file(s), "
          f"{len(baseline_candidates)} baseline (individual model) file(s)")

    if args.split != "all":
        before = len(ensemble_candidates)
        # Shetland ensemble rows are always split='full' (never dev/test) -
        # keep them regardless of what --split was requested for the other
        # datasets, instead of silently excluding Shetland's one-shot result
        ensemble_candidates = [
            r for r in ensemble_candidates
            if r["split"] == args.split or r["dataset"] == "shetland"
        ]
        print(f"Filtered ensemble rows to split='{args.split}': {len(ensemble_candidates)}/{before} remain "
              f"(baseline rows are never split-filtered; Shetland rows are always kept regardless of split)")

        print(f"\nRecomputing baseline (individual model) WER/severity restricted to "
              f"split='{args.split}' - so they're compared fairly against dev-only ensembles "
              f"(Shetland baselines always use 'full' - see note above if applicable):")
        baseline_candidates = [recompute_baseline_for_split(r, args.split) for r in baseline_candidates]

    ensemble_rows = dedupe_rows(ensemble_candidates)
    baseline_rows = dedupe_rows(baseline_candidates)
    print(f"After deduping: {len(ensemble_rows)} ensemble + {len(baseline_rows)} baseline result(s)")

    print_pool_diagnostics(baseline_rows, "baseline rows used for in-domain micro-average")
    print_pool_diagnostics(ensemble_rows, "ensemble rows used for in-domain micro-average")

    if not ensemble_rows and not baseline_rows:
        print("\nNo matching result files found - check --roots and --split.")
        return

    by_dataset = defaultdict(list)
    for r in ensemble_rows:
        by_dataset[r["dataset"]].append(r)

    # find the best baseline PER DATASET (not by average - the baseline
    # results are mixed, no single model wins everywhere) and inject it
    # into each dataset's own table, labeled distinctly, so you can see
    # directly whether ensemble techniques actually rank above it
    per_dataset_best_baseline = best_baseline_per_dataset(baseline_rows)
    for ds, best_row in per_dataset_best_baseline.items():
        labeled = {**best_row, "approach": f"[baseline] {best_row['approach']}"}
        by_dataset[ds].append(labeled)

    # Shetland is special: only ONE technique (your locked winner) was
    # ever run there, per the dev/test/Shetland discipline - so instead
    # of showing just the single best baseline, show ALL FOUR baseline
    # models alongside it for the full picture in one table.
    if "shetland" in by_dataset:
        shetland_baselines = [r for r in baseline_rows if r["dataset"] == "shetland"]
        # remove the single-best-baseline row already added above for
        # shetland (already labeled "[baseline] ..."), replace with all 4
        by_dataset["shetland"] = [
            r for r in by_dataset["shetland"] if not r["approach"].startswith("[baseline]")
        ]
        for r in shetland_baselines:
            by_dataset["shetland"].append({**r, "approach": f"[baseline] {r['approach']}"})

    dataset_headers = ["Technique", "Mean Severity", "Corpus WER", "N scored"]
    dataset_aligns = ["<", ">", ">", ">"]
    combined_headers = ["Technique"]
    for ds in DATASET_ORDER:
        combined_headers += [f"{ds} Sev", f"{ds} WER"]
    combined_headers += ["Avg Sev", "Avg WER"]
    combined_aligns = ["<"] + [">"] * (len(combined_headers) - 1)

    markdown_sections = []

    # ── Table set 1: per-dataset tables (ensembles + that dataset's best baseline) ──
    all_datasets = DATASET_ORDER + [ds for ds in by_dataset if ds not in DATASET_ORDER]
    for ds in all_datasets:
        if ds not in by_dataset:
            continue
        print(f"\n{'=' * 60}")
        print(f"  {ds.upper()}")
        print(f"{'=' * 60}")
        table_rows = build_dataset_table_rows(by_dataset[ds])

        if ds == "shetland":
            # highlight the best baseline (lowest severity among the 4
            # [baseline]-labeled rows) and naive (the winner) - matching
            # the same "* BEST" marker convention used elsewhere in your
            # pipeline (selector_ablation.py's comparison table)
            baseline_row_indices = [i for i, r in enumerate(table_rows) if r[0].startswith("[baseline]")]
            if baseline_row_indices:
                # table_rows is already sorted by severity ascending, so
                # the first baseline row encountered is the best one
                best_idx = baseline_row_indices[0]
                table_rows[best_idx] = list(table_rows[best_idx])
                table_rows[best_idx][0] += "  * BEST BASELINE"
            for i, r in enumerate(table_rows):
                if r[0] == "naive":
                    table_rows[i] = list(table_rows[i])
                    table_rows[i][0] += "  * WINNER"

        print_table(dataset_headers, table_rows, dataset_aligns)
        if args.markdown_out:
            markdown_sections.append(f"## {ds}\n\n" + build_markdown_table(dataset_headers, table_rows))

    # ── Table: best baseline vs best ensemble, head to head, PER dataset ──
    print(f"\n{'=' * 60}")
    print(f"  BEST BASELINE vs BEST ENSEMBLE (per dataset)")
    print(f"{'=' * 60}")
    per_dataset_best_ensemble = best_ensemble_per_dataset(ensemble_rows)
    h2h_headers = ["Dataset", "Best Baseline", "Base Sev", "Base WER",
                   "Best Ensemble", "Ens Sev", "Ens WER", "Winner"]
    h2h_aligns = ["<", "<", ">", ">", "<", ">", ">", "<"]
    h2h_rows = build_head_to_head_rows(per_dataset_best_baseline, per_dataset_best_ensemble)
    print_table(h2h_headers, h2h_rows, h2h_aligns)
    if args.markdown_out:
        markdown_sections.append("## Best baseline vs best ensemble (per dataset)\n\n" +
                                   build_markdown_table(h2h_headers, h2h_rows))

    # ── Table 2: ALL baseline (individual ASR) models, combined across datasets ──
    print(f"\n{'=' * 60}")
    print(f"  BASELINE MODELS - MACRO-AVERAGE (each dataset weighted equally)")
    print(f"{'=' * 60}")
    baseline_combined_rows = build_combined_table_rows(baseline_rows)
    print_table(combined_headers, baseline_combined_rows, combined_aligns)
    if args.markdown_out:
        markdown_sections.append("## Baseline models - macro-average (combined across datasets)\n\n" +
                                   build_markdown_table(combined_headers, baseline_combined_rows))

    # ── Table 2b: baseline models, POOLED (micro-average) across datasets ──
    print(f"\n{'=' * 60}")
    print(f"  BASELINE MODELS - POOLED / MICRO-AVERAGE (every sample weighted equally)")
    print(f"{'=' * 60}")
    pooled_headers = ["Technique", "Pooled Severity", "Pooled WER", "N (total samples)"]
    pooled_aligns = ["<", ">", ">", ">"]
    baseline_pooled_rows = build_pooled_table_rows(baseline_rows, datasets=DATASET_ORDER)
    print_table(pooled_headers, baseline_pooled_rows, pooled_aligns)
    if args.markdown_out:
        markdown_sections.append("## Baseline models - pooled/micro-average (every sample weighted equally)\n\n" +
                                   build_markdown_table(pooled_headers, baseline_pooled_rows))

    # ── Table 3: ALL ensemble techniques, combined across datasets ──
    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE TECHNIQUES - MACRO-AVERAGE (each dataset weighted equally)")
    print(f"{'=' * 60}")
    ensemble_combined_rows = build_combined_table_rows(ensemble_rows)
    print_table(combined_headers, ensemble_combined_rows, combined_aligns)
    if args.markdown_out:
        markdown_sections.append("## Ensemble techniques - macro-average (combined across datasets)\n\n" +
                                   build_markdown_table(combined_headers, ensemble_combined_rows))

    # ── Table 3b: ensemble techniques, POOLED (micro-average) across datasets ──
    print(f"\n{'=' * 60}")
    print(f"  ENSEMBLE TECHNIQUES - POOLED / MICRO-AVERAGE (every sample weighted equally)")
    print(f"{'=' * 60}")
    ensemble_pooled_rows = build_pooled_table_rows(ensemble_rows, datasets=DATASET_ORDER)
    print_table(pooled_headers, ensemble_pooled_rows, pooled_aligns)
    if args.markdown_out:
        markdown_sections.append("## Ensemble techniques - pooled/micro-average (every sample weighted equally)\n\n" +
                                   build_markdown_table(pooled_headers, ensemble_pooled_rows))

    # ── Table 4: FINAL - top 2 ensembles vs top 2 baselines, head to head ──
    print(f"\n{'=' * 60}")
    print(f"  FINAL: TOP {args.top_ensembles} ENSEMBLE(S) vs TOP {args.top_baselines} BASELINE(S)")
    print(f"{'=' * 60}")
    top_ensemble_rows = pick_top_n(ensemble_rows, args.top_ensembles, label="ensemble technique")
    top_baseline_rows = pick_top_n(baseline_rows, args.top_baselines, label="baseline model")
    top_baseline_rows = [
        {**r, "approach": f"[baseline] {r['approach']}"} for r in top_baseline_rows
    ]
    final_rows = build_combined_table_rows(top_ensemble_rows + top_baseline_rows)
    print_table(combined_headers, final_rows, combined_aligns)
    if args.markdown_out:
        markdown_sections.append(f"## Final: top {args.top_ensembles} ensemble(s) vs "
                                   f"top {args.top_baselines} baseline(s)\n\n" +
                                   build_markdown_table(combined_headers, final_rows))

    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write("\n\n".join(markdown_sections) + "\n")
        print(f"\nSaved markdown version -> {args.markdown_out}")


if __name__ == "__main__":
    main()