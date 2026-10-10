import json
import os

EXPECTED_SIZES = {
    "commonvoice_test": 196,
    "edacc_test": 58,
    "english_dialects_test": 744,
    "shetland_full": 100,
}

JOBS = {
    "5model": "writeup_results/grid/unanchored_fusion_naive_5model/unanchored_fusion_naive_5model_{d}_gemma4_{s}.json",
    "replace_whisperx": "writeup_results/grid/unanchored_fusion_naive_replace_whisperx/unanchored_fusion_naive_replace_whisperx_{d}_gemma4_{s}.json",
}

RUNS = [
    ("commonvoice", "test"),
    ("edacc", "test"),
    ("english_dialects", "test"),
    ("shetland", "full"),
]

for label, pattern in JOBS.items():
    print(f"\n{label}:")
    for d, s in RUNS:
        path = pattern.format(d=d, s=s)
        expected = EXPECTED_SIZES[f"{d}_{s}"]
        if not os.path.exists(path):
            print(f"  {d} ({s}): NOT STARTED")
            continue
        data = json.load(open(path))
        samples = data.get("samples", [])
        n_done = len(samples)
        n_scored = sum(1 for x in samples if x.get("severity") is not None)
        corpus_wer = data.get("corpus_wer")
        mean_sev = data.get("mean_severity")
        status = "COMPLETE" if n_done >= expected else f"{n_done}/{expected}"
        wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
        sev_str = f"{mean_sev:.3f}" if mean_sev is not None else "-"
        print(f"  {d} ({s}): {status}  scored={n_scored}  wer={wer_str}  severity={sev_str}")
