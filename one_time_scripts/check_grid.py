import json
import glob

patterns = [
    "writeup_results/grid/selection_naive/*.json",
    "writeup_results/grid/selection_context_v1/*.json",
    "writeup_results/grid/selection_context_v2/*.json",
    "writeup_results/grid/unanchored_fusion_context_v1/*.json",
    "writeup_results/grid/unanchored_fusion_context_v2/*.json",
    "writeup_results/grid/anchored_correction_naive/*.json",
]

for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        approach = d.get("approach", "?")
        dataset = d.get("dataset", "?")
        selector = d.get("selector", "?")
        wer = d.get("corpus_wer")
        sev = d.get("mean_severity")
        n = d.get("num_samples", "?")
        wer_str = f"{wer*100:.2f}%" if wer is not None else "-"
        sev_str = f"{sev:.3f}" if sev is not None else "PENDING"
        print(f"{approach:<28} {dataset:<18} {selector:<8} WER={wer_str:>8}  Severity={sev_str:>9}  N={n}")
