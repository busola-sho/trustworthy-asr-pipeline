import json

for approach, path in [
    ("unanchored_fusion_naive", "writeup_results/grid_calib_fixed/unanchored_fusion_naive"),
    ("anchored_correction_v1", "writeup_results/grid_calib_fixed/anchored_correction_v1"),
]:
    print(f"\n{approach}:")
    for d in ["commonvoice", "edacc", "english_dialects"]:
        data = json.load(open(f"{path}/{approach}_{d}_gemma4_test.json"))
        print(f"  {d}: severity={data.get('mean_severity')}  wer={data.get('corpus_wer')}  N={data.get('num_samples')}")
