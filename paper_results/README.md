# Saved paper results

`01_model_combination/` contains the saved four-model comparison, guidance and
post-edit results. `02_sentence_confidence/` contains the original five-model
confidence results, tables, figures and calibrators. These saved files are unchanged.
The active four-model confidence pipeline writes elsewhere, under `runs/`.

`reporting/` is the supported report entry point. Its `combination/` and
`confidence/` packages contain the writers and shared scoring/plotting helpers.
Functions receive input and output paths as arguments. Importing them does not
change another module’s folders or split assignments.

Install the reporting environment from the repository root:

```bash
pip install -e '.[reporting]'
```

To reproduce reports later, choose a new output directory:

```bash
python -m asr_pipeline report \
  --saved-results-dir paper_results --output-dir /tmp/asr-paper-report \
  --splits-file paper_results/splits.json \
  --input-files paper_results/input_files.json \
  --original-results-dir paper_results --reports tables confidence-figures
```

This expects 40 output files and checks that this run created each one.
For all 44 supported outputs, add `downstream` to `--reports` and supply:

```text
--police-individual paper_results/supplemental/data/police_individual.json
--police-fusion paper_results/supplemental/data/police_fusion.json
--shetland-anchored paper_results/supplemental/data/shetland_anchored.json
```

Reporting can fit small statistical calibrators from saved scores. It never runs
ASR or LLM prediction.
Newly generated CSV/JSON/TXT files are compared with saved references (absolute
numeric tolerance `1e-12`; grid source-path headers may differ). PNG/pickle files
are checked as new/nonempty, not assumed byte-identical. `run-manifest.json` lists the output files and checks. The old `spearman_correlations.png` has no current producer
and is preserved without claiming reproduction.

`input_files.json` identifies 122 saved paper input files and six extra
source files. The `data/` folders are ignored because release permission has not been established.
A fresh clone needs these inputs supplied separately.
No participant records or original operational audio/transcripts are included.
Missing paper analyses remain deferred. Earlier history notes and
unused reporting/experiment scripts are preserved in the backup branch and local
archive outside the active code.
