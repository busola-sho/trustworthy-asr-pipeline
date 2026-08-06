import json
import os

EXPECTED_SIZES = {
    "commonvoice": 680,
    "edacc": 198,
    "english_dialects": 2543,
    "shetland": 100,
}

for dataset, expected in EXPECTED_SIZES.items():
    path = f"writeup_results/benchmarks/main/whisper_ft_chunked_{dataset}.json"
    if not os.path.exists(path):
        print(f"{dataset}: NOT STARTED")
        continue
    data = json.load(open(path))
    samples = data.get("samples", [])
    n_done = len(samples)
    n_scored = sum(1 for s in samples if s.get("severity") is not None)
    corpus_wer = data.get("corpus_wer")
    mean_sev = data.get("mean_severity")
    status = "COMPLETE" if n_done >= expected else f"{n_done}/{expected}"
    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    sev_str = f"{mean_sev:.3f}" if mean_sev is not None else "-"
    print(f"{dataset}: {status}  scored={n_scored}  wer={wer_str}  severity={sev_str}")

