"""
sample_for_human_eval_v2.py

Samples 25 audio clips total, stratified equally across 3 datasets
(~8 per dataset), then pulls all 4 model outputs for each clip.

Output: 100 rows (25 clips × 4 models)
  - human_eval_samples.json  — source of truth
  - human_eval_samples.xlsx  — annotation template (you fill human_verdict)

Quadrant stratification per dataset:
  Q1: low WER  + meaning_altering=True   (WER/MAR decoupling — most interesting)
  Q2: high WER + meaning_altering=True   (clear failures)
  Q3: low WER  + meaning_altering=False  (clean baseline)
  Q4: high WER + meaning_altering=False  (surface errors, no semantic damage)

WER threshold: 0.3
"""

import json
import random
import os
from collections import defaultdict

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ── Configuration ──────────────────────────────────────────────────────────────

BENCHMARKS_DIR = "benchmarks"
OUTPUT_JSON    = "human_eval_samples.json"
OUTPUT_XLSX    = "human_eval_samples.xlsx"
SEED           = 42
WER_THRESHOLD  = 0.3

# 25 clips split across 3 datasets as equally as possible
DATASET_COUNTS = {
    "commonvoice":      8,
    "english_dialects": 8,
    "edacc":            9,
}

CANONICAL_FILES = {
    ("parakeet",  "commonvoice"):       "parakeet_commonvoice_20260524_150129.json",
    ("parakeet",  "edacc"):             "parakeet_edacc_20260525_184006.json",
    ("parakeet",  "english_dialects"):  "parakeet_english_dialects_20260524_234807.json",
    ("qwen",      "commonvoice"):       "qwen_commonvoice_20260524_153426.json",
    ("qwen",      "edacc"):             "qwen_edacc_20260525_204314.json",
    ("qwen",      "english_dialects"):  "qwen_english_dialects_20260525_000627.json",
    ("wav2vec2",  "commonvoice"):       "wav2vec2_commonvoice_20260526_053757.json",
    ("wav2vec2",  "edacc"):             "wav2vec2_edacc_20260525_213534.json",
    ("wav2vec2",  "english_dialects"):  "wav2vec2_english_dialects_20260526_073439.json",
    ("whisper",   "commonvoice"):       "whisper_commonvoice_20260524_083930.json",
    ("whisper",   "edacc"):             "whisper_edacc_20260525_184504.json",
    ("whisper",   "english_dialects"):  "whisper_english_dialects_20260525_110315.json",
}

MODELS   = ["qwen", "whisper", "parakeet", "wav2vec2"]
DATASETS = ["commonvoice", "english_dialects", "edacc"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def is_valid(s):
    if "IGNORE_TIME_SEGMENT_IN_SCORING" in s.get("ref", ""):
        return False
    if not s.get("ref", "").strip() or not s.get("hyp", "").strip():
        return False
    if s.get("sample_WER") is None or s.get("meaning_altering") is None:
        return False
    return True

def quadrant(s):
    wer = s["sample_WER"]
    ma  = s["meaning_altering"]
    if   wer <= WER_THRESHOLD and ma:      return "Q1_lowWER_MA"
    elif wer >  WER_THRESHOLD and ma:      return "Q2_highWER_MA"
    elif wer <= WER_THRESHOLD and not ma:  return "Q3_lowWER_noMA"
    else:                                  return "Q4_highWER_noMA"

def stratified_sample(valid_samples, n, seed=42):
    """
    Sample n indices from valid_samples, spread across 4 quadrants.
    Returns positions into valid_samples list.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, s in enumerate(valid_samples):
        buckets[quadrant(s)].append(i)

    quota = n // 4
    selected = []
    remainder = []

    for q in ["Q1_lowWER_MA", "Q2_highWER_MA", "Q3_lowWER_noMA", "Q4_highWER_noMA"]:
        bucket = list(buckets[q])
        rng.shuffle(bucket)
        selected.extend(bucket[:quota])
        remainder.extend(bucket[quota:])

    still_needed = n - len(selected)
    if still_needed > 0:
        rng.shuffle(remainder)
        selected.extend(remainder[:still_needed])

    return selected

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Step 1: for each dataset, select clip indices using qwen as reference
    # (qwen is the best performer — most complete, fewest empty hyps)
    dataset_clip_indices = {}

    for dataset, n_clips in DATASET_COUNTS.items():
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[("qwen", dataset)])
        data = load_json(path)
        all_samples = data["samples"]

        valid_pairs = [(i, s) for i, s in enumerate(all_samples) if is_valid(s)]
        valid_indices = [i for i, _ in valid_pairs]
        valid_samples = [s for _, s in valid_pairs]

        if len(valid_samples) < n_clips:
            print(f"WARNING: {dataset} only has {len(valid_samples)} valid samples, using all.")
            n_clips = len(valid_samples)

        selected_positions = stratified_sample(valid_samples, n_clips)
        selected_original  = [valid_indices[p] for p in selected_positions]
        dataset_clip_indices[dataset] = selected_original

        q_counts = defaultdict(int)
        for p in selected_positions:
            q_counts[quadrant(valid_samples[p])] += 1
        print(f"{dataset}: {n_clips} clips selected — {dict(q_counts)}")

    # Step 2: build 100 rows (25 clips × 4 models)
    rows = []
    clip_counter = 0

    for dataset in DATASETS:
        indices = dataset_clip_indices[dataset]

        for idx in indices:
            clip_id = f"clip_{clip_counter:03d}"
            clip_counter += 1

            for model in MODELS:
                path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, dataset)])
                data = load_json(path)
                samples = data["samples"]

                if idx >= len(samples):
                    print(f"WARNING: index {idx} out of range for {model}/{dataset}")
                    continue

                s = samples[idx]
                rows.append({
                    "clip_id":          clip_id,           # same across 4 models
                    "sample_index":     idx,               # position in original JSON
                    "dataset":          dataset,
                    "model":            model,
                    "quadrant":         quadrant(s),       # based on qwen; may differ per model
                    "ref":              s["ref"],
                    "hyp":              s["hyp"],
                    "sample_wer":       round(s["sample_WER"], 4),
                    "gpt4o_verdict":    s["meaning_altering"],
                    "human_verdict":    None,              # TRUE/FALSE — fill manually
                    "human_notes":      None,
                    "selene_verdict":   None,              # filled programmatically
                    "ollama_verdict":   None,
                })

    # Step 3: write JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_JSON}")

    # Step 4: write Excel
    if HAS_OPENPYXL:
        write_excel(rows)
    else:
        print("openpyxl not installed — run: pip install openpyxl")

# ── Excel writer ───────────────────────────────────────────────────────────────

Q_COLOURS = {
    "Q1_lowWER_MA":    "FFD700",  # gold
    "Q2_highWER_MA":   "FF6B6B",  # red
    "Q3_lowWER_noMA":  "90EE90",  # green
    "Q4_highWER_noMA": "ADD8E6",  # blue
}

MODEL_COLOURS = {
    "qwen":     "E8F5E9",
    "whisper":  "E3F2FD",
    "parakeet": "FFF3E0",
    "wav2vec2": "FCE4EC",
}

def write_excel(rows):
    wb = openpyxl.Workbook()

    # ── Sheet 1: All 100 rows, sorted by clip_id then model ───────────────────
    ws = wb.active
    ws.title = "Annotations"

    headers = [
        "clip_id", "dataset", "model", "quadrant",
        "sample_wer", "gpt4o_verdict",
        "human_verdict",    # fill: TRUE or FALSE
        "human_notes",
        "selene_verdict", "ollama_verdict",
        "ref", "hyp",
    ]
    ws.append(headers)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F4F4F")
    for cell in ws[1]:
        cell.font  = header_font
        cell.fill  = header_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    for row in rows:
        ws.append([
            row["clip_id"],
            row["dataset"],
            row["model"],
            row["quadrant"],
            row["sample_wer"],
            row["gpt4o_verdict"],
            "",   # human_verdict
            "",   # human_notes
            "",   # selene_verdict
            "",   # ollama_verdict
            row["ref"],
            row["hyp"],
        ])
        r = ws.max_row
        # colour quadrant cell
        ws.cell(r, 4).fill = PatternFill("solid", fgColor=Q_COLOURS.get(row["quadrant"], "FFFFFF"))
        # colour model cell
        ws.cell(r, 3).fill = PatternFill("solid", fgColor=MODEL_COLOURS.get(row["model"], "FFFFFF"))
        # wrap ref/hyp
        ws.cell(r, 11).alignment = Alignment(wrap_text=True)
        ws.cell(r, 12).alignment = Alignment(wrap_text=True)

    col_widths = [10, 18, 12, 20, 10, 14, 14, 25, 14, 14, 55, 55]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30

    # ── Sheet 2: per-clip view — one clip per block, all 4 models together ────
    ws2 = wb.create_sheet("By Clip")
    ws2.append(["clip_id", "dataset", "ref", "model", "hyp", "sample_wer",
                 "gpt4o_verdict", "human_verdict", "human_notes"])
    ws2[1][0].font = Font(bold=True)

    clip_groups = defaultdict(list)
    for row in rows:
        clip_groups[row["clip_id"]].append(row)

    for clip_id, clip_rows in clip_groups.items():
        ref = clip_rows[0]["ref"]
        dataset = clip_rows[0]["dataset"]
        for i, row in enumerate(clip_rows):
            ws2.append([
                clip_id if i == 0 else "",
                dataset if i == 0 else "",
                ref     if i == 0 else "",
                row["model"],
                row["hyp"],
                row["sample_wer"],
                row["gpt4o_verdict"],
                "",  # human_verdict
                "",  # human_notes
            ])
            r = ws2.max_row
            ws2.cell(r, 4).fill = PatternFill("solid", fgColor=MODEL_COLOURS.get(row["model"], "FFFFFF"))
            ws2.cell(r, 5).alignment = Alignment(wrap_text=True)
        # blank separator row between clips
        ws2.append([""] * 9)

    clip_col_widths = [10, 18, 55, 12, 55, 10, 14, 14, 25]
    for i, w in enumerate(clip_col_widths, 1):
        ws2.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws2.freeze_panes = "A2"

    # ── Sheet 3: agreement summary template ───────────────────────────────────
    ws3 = wb.create_sheet("Agreement Summary")
    ws3.append(["metric", "qwen", "whisper", "parakeet", "wav2vec2", "all_models"])
    for metric in [
        "human_vs_gpt4o agreement (%)",
        "human_vs_selene agreement (%)",
        "human_vs_ollama agreement (%)",
        "gpt4o_vs_selene agreement (%)",
        "gpt4o_vs_ollama agreement (%)",
        "MAR — human",
        "MAR — gpt4o",
        "MAR — selene",
        "MAR — ollama",
    ]:
        ws3.append([metric] + [""] * 5)

    for cell in ws3[1]:
        cell.font = Font(bold=True)

    wb.save(OUTPUT_XLSX)
    print(f"Wrote annotation template to {OUTPUT_XLSX}")
    print("\nSheets:")
    print("  'Annotations' — flat 100-row table, fill human_verdict column")
    print("  'By Clip'     — grouped by clip, all 4 models together (easier for annotation)")
    print("  'Agreement Summary' — fill after running LLM judges")

if __name__ == "__main__":
    main()