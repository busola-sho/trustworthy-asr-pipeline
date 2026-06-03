"""
summarise_all.py

Generates a comprehensive markdown summary of all benchmark results:
1. Individual model benchmarks (WER + GPT-4o MAR + Qwen P2 MAR)
2. LibriSpeech verification (WER vs published)
3. Naive combination results (WER + MAR by selector/judge)

Output: benchmarks/full_summary.md
"""

import json
import glob
import os
from collections import defaultdict

BENCHMARKS_DIR = "benchmarks"
OUTPUT_PATH    = os.path.join(BENCHMARKS_DIR, "full_summary.md")

CANONICAL_MODELS = {
    "qwen":     "Qwen3-ASR-1.7B",
    "whisper":  "Whisper large-v3",
    "parakeet": "Parakeet CTC 1.1B",
    "wav2vec2": "wav2vec2-large-960h",
}

CANONICAL_DATASETS = {
    "common_voice":                  "CommonVoice Scottish",
    "english_dialects_scots":        "English Dialects Scottish",
    "edinburgh_international_accents": "EdAcc Scottish",
}

PUBLISHED_WER = {
    "whisper":  2.7,
    "wav2vec2": 1.8,
    "parakeet": 1.83,
    "qwen3asr": 1.63,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def pct(v):
    if v is None:
        return "—"
    return f"{round(v * 100, 2)}%"

def detect_model_key(model_name: str) -> str:
    model_name = model_name.lower()
    if "qwen" in model_name:
        return "qwen"
    if "whisper" in model_name:
        return "whisper"
    if "parakeet" in model_name:
        return "parakeet"
    if "wav2vec2" in model_name:
        return "wav2vec2"
    return model_name

def detect_dataset_key(dataset_name: str) -> str:
    if "common_voice" in dataset_name or "commonvoice" in dataset_name:
        return "CommonVoice Scottish"
    if "english_dialects" in dataset_name:
        return "English Dialects Scottish"
    if "edinburgh" in dataset_name or "edacc" in dataset_name:
        return "EdAcc Scottish"
    return dataset_name

def load_json(path):
    with open(path) as f:
        return json.load(f)

# ── Section 1: Individual model benchmarks ────────────────────────────────────

def build_individual_section():
    # load all canonical benchmark files
    canonical_files = {
        ("qwen",     "CommonVoice Scottish"):       "qwen_commonvoice_20260524_153426.json",
        ("qwen",     "English Dialects Scottish"):  "qwen_english_dialects_20260525_000627.json",
        ("qwen",     "EdAcc Scottish"):             "qwen_edacc_20260525_204314.json",
        ("whisper",  "CommonVoice Scottish"):       "whisper_commonvoice_20260524_083930.json",
        ("whisper",  "English Dialects Scottish"):  "whisper_english_dialects_20260525_110315.json",
        ("whisper",  "EdAcc Scottish"):             "whisper_edacc_20260525_184504.json",
        ("parakeet", "CommonVoice Scottish"):       "parakeet_commonvoice_20260524_150129.json",
        ("parakeet", "English Dialects Scottish"):  "parakeet_english_dialects_20260524_234807.json",
        ("parakeet", "EdAcc Scottish"):             "parakeet_edacc_20260525_184006.json",
        ("wav2vec2", "CommonVoice Scottish"):       "wav2vec2_commonvoice_20260526_053757.json",
        ("wav2vec2", "English Dialects Scottish"):  "wav2vec2_english_dialects_20260526_073439.json",
        ("wav2vec2", "EdAcc Scottish"):             "wav2vec2_edacc_20260525_213534.json",
    }

    # build results table: {dataset: {model: {wer, gpt4o_mar, qwen_mar, n}}}
    results = defaultdict(dict)

    for (model_key, dataset), filename in canonical_files.items():
        path = os.path.join(BENCHMARKS_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: missing {filename}")
            continue

        data = load_json(path)
        samples = data.get("samples", [])

        # filter valid samples
        valid = [
            s for s in samples
            if "IGNORE_TIME_SEGMENT_IN_SCORING" not in s.get("ref", "")
            and s.get("sample_WER") is not None
        ]

        gpt4o_mar = sum(1 for s in valid if s.get("meaning_altering")) / len(valid) if valid else None
        qwen_mar  = sum(1 for s in valid if s.get("qwen_verdict")) / len(valid) if valid else None

        results[dataset][model_key] = {
            "wer":       data.get("corpus_wer"),
            "gpt4o_mar": gpt4o_mar,
            "qwen_mar":  qwen_mar,
            "n":         len(valid),
        }

    datasets_order = ["CommonVoice Scottish", "English Dialects Scottish", "EdAcc Scottish"]
    models_order   = ["qwen", "whisper", "parakeet", "wav2vec2"]
    model_labels   = {
        "qwen":     "Qwen3-ASR-1.7B",
        "whisper":  "Whisper large-v3",
        "parakeet": "Parakeet CTC 1.1B",
        "wav2vec2": "wav2vec2-large-960h",
    }

    md = "## 1. Individual Model Benchmarks\n\n"
    md += "> MAR computed under two judges: GPT-4o Prompt 2 and Qwen 2.5 7B Prompt 2.\n"
    md += "> Qwen P2 selected as production judge (88.9% human agreement, 8.1% false negative rate).\n\n"

    for dataset in datasets_order:
        if dataset not in results:
            continue
        md += f"### {dataset}\n\n"
        md += "| Model | N | WER | MAR (GPT-4o P2) | MAR (Qwen P2) |\n"
        md += "|-------|---|-----|-----------------|---------------|\n"

        rows = [(m, results[dataset].get(m, {})) for m in models_order]
        best_wer = min((r["wer"] for _, r in rows if r.get("wer") is not None), default=None)

        for model_key, r in rows:
            if not r:
                continue
            label = model_labels[model_key]
            wer_str = pct(r.get("wer"))
            if r.get("wer") == best_wer:
                wer_str = f"**{wer_str}**"
            md += f"| {label} | {r.get('n', '—')} | {wer_str} | {pct(r.get('gpt4o_mar'))} | {pct(r.get('qwen_mar'))} |\n"
        md += "\n"

    return md

# ── Section 2: LibriSpeech verification ───────────────────────────────────────

def build_librispeech_section():
    ls_files = glob.glob(os.path.join(BENCHMARKS_DIR, "librispeech_*.json"))

    # pick most recent per model
    model_results = {}
    for path in sorted(ls_files):
        fname = os.path.basename(path)
        for key in ["whisper", "parakeet", "wav2vec2", "qwen3asr"]:
            if key in fname:
                model_results[key] = path

    if not model_results:
        return "## 2. LibriSpeech Verification\n\n> No results found.\n\n"

    md = "## 2. LibriSpeech test-clean Verification\n\n"
    md += "> Verifies pipeline implementation against published benchmarks.\n"
    md += "> Source: Open ASR Leaderboard (Srivastav et al., 2025; arXiv:2510.06961)\n\n"
    md += "| Model | N | Our WER | Published WER | Gap |\n"
    md += "|-------|---|---------|---------------|-----|\n"

    model_labels = {
        "whisper":  "Whisper large-v3",
        "parakeet": "Parakeet CTC 1.1B",
        "wav2vec2": "wav2vec2-large-960h",
        "qwen3asr": "Qwen3-ASR-1.7B",
    }

    for model_key in ["qwen3asr", "whisper", "parakeet", "wav2vec2"]:
        if model_key not in model_results:
            continue
        data = load_json(model_results[model_key])
        our_wer   = data.get("corpus_wer")
        published = PUBLISHED_WER.get(model_key)
        n         = data.get("num_samples", "—")
        label     = model_labels[model_key]

        if our_wer is not None and published is not None:
            gap = round(our_wer * 100 - published, 2)
            gap_str = f"+{gap}pp" if gap > 0 else f"{gap}pp"
        else:
            gap_str = "—"

        our_wer_str = pct(our_wer) if our_wer else "pending"
        md += f"| {label} | {n} | {our_wer_str} | {published}% | {gap_str} |\n"

    md += "\n"
    return md

# ── Section 3: Naive combination ──────────────────────────────────────────────

def build_naive_section():
    naive_map = {
        "naive_commonvoice_gpt4o.json":            ("CommonVoice", "GPT-4o",  "GPT-4o P2"),
        "naive_commonvoice_qwen.json":             ("CommonVoice", "Qwen",    "Qwen P2"),
        "naive_commonvoice_llamasel_qwenjud.json": ("CommonVoice", "Llama",   "Qwen P2"),
        "naive_commonvoice_qwensel_llamajud.json": ("CommonVoice", "Qwen",    "Llama P2"),
        "naive_commonvoice_llamasel_llamajud.json":("CommonVoice", "Llama",   "Llama P2"),
        "naive_edacc_gpt4o.json":                  ("EdAcc",       "GPT-4o",  "GPT-4o P2"),
        "naive_edacc_qwen.json":                   ("EdAcc",       "Qwen",    "Qwen P2"),
        "naive_edacc_llamasel_qwenjud.json":        ("EdAcc",       "Llama",   "Qwen P2"),
        "naive_edacc_qwensel_llamajud.json":        ("EdAcc",       "Qwen",    "Llama P2"),
        "naive_edacc_llamasel_llamajud.json":       ("EdAcc",       "Llama",   "Llama P2"),
    }

    rows = []
    for fname, (dataset, selector, judge) in naive_map.items():
        path = os.path.join(BENCHMARKS_DIR, fname)
        if not os.path.exists(path):
            continue
        data = load_json(path)
        wer = data.get("corpus_wer")
        mar = data.get("meaning_alteration_rate")
        n   = data.get("num_samples", "—")
        rows.append({
            "dataset":  dataset,
            "selector": selector,
            "judge":    judge,
            "wer":      wer,
            "mar":      mar,
            "n":        n,
        })

    if not rows:
        return "## 3. Naive Combination Baseline\n\n> No results found.\n\n"

    md = "## 3. Naive Combination Baseline\n\n"
    md += "> All 4 model transcriptions passed to an LLM selector to produce one combined transcript.\n"
    md += "> Compared against best individual model (Qwen3-ASR).\n\n"

    for dataset in ["CommonVoice", "EdAcc"]:
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        if not dataset_rows:
            continue

        md += f"### {dataset}\n\n"
        md += "| System | Selector | Judge | N | WER | MAR |\n"
        md += "|--------|----------|-------|---|-----|-----|\n"

        # best individual model row first
        best_wer = {"CommonVoice": 0.1884, "EdAcc": 0.1692}
        best_mar = {"CommonVoice": 0.7368, "EdAcc": 0.2475}  # Qwen P2 MAR
        md += f"| **Best single (Qwen3-ASR)** | — | Qwen P2 | — | **{pct(best_wer[dataset])}** | {pct(best_mar[dataset])} |\n"

        for r in sorted(dataset_rows, key=lambda x: x["wer"] or 99):
            md += f"| Naive | {r['selector']} | {r['judge']} | {r['n']} | {pct(r['wer'])} | {pct(r['mar'])} |\n"

        md += "\n"

    md += "> **Key finding:** No naive combination beats the best individual model on WER.\n"
    md += "> Llama judge consistently under-reports MAR due to high false negative rate (25.3% from calibration).\n\n"

    return md

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building full summary...")

    md  = "# Trustworthy ASR Pipeline — Results Summary\n\n"
    md += "> Generated by `scripts/summarise_all.py`\n\n"
    md += "---\n\n"

    print("  Section 1: Individual model benchmarks...")
    md += build_individual_section()
    md += "---\n\n"

    print("  Section 2: LibriSpeech verification...")
    md += build_librispeech_section()
    md += "---\n\n"

    print("  Section 3: Naive combination...")
    md += build_naive_section()

    with open(OUTPUT_PATH, "w") as f:
        f.write(md)

    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()