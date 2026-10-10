import json

data = json.load(open("writeup_results/clean_grid/selection_naive/selection_naive_edacc_gemma4_dev.json"))
samples = data.get("samples", [])
unscored = [s for s in samples if s.get("severity") is None]

print(f"Total unscored: {len(unscored)}")
for s in unscored:
    print(f"  dataset_index={s.get('dataset_index')}  ref={str(s.get('ref'))[:40]!r}  "
          f"skipped={s.get('skipped')}  skip_reason={s.get('skip_reason')}  "
          f"error={s.get('error')}  error_reason={s.get('error_reason')}")
