# Datasets and preparation

The bundled examples need no corpus download. Download the full research datasets separately:

| Dataset used | Actual source and access | Loader selection / filtering |
|---|---|---|
| Common Voice **Spontaneous Speech Scots** (`sco`), `sps-corpus-3.0-2026-03-09` | [Release metadata](https://github.com/common-voice/cv-dataset/blob/main/datasets/spontaneous-speech/sps-corpus-3.0-2026-03-09.json); obtain the Scots **spontaneous**, not scripted English, release through [Mozilla Data Collective](https://mozilladatacollective.com/). Sign in and follow the dataset's download/access terms. | Local `ss-corpus-sco.tsv` plus `audios/`; drop rows missing `transcription`, retain TSV order. The local release has 680 transcribed rows. No additional accent, duration or upstream train/dev/test filter is present in the loader. |
| EdAcc, Scottish English subset | [edinburghcstr/edacc](https://huggingface.co/datasets/edinburghcstr/edacc), cached revision `d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66`; download using Hugging Face Datasets. | Load upstream **test**; retain exact `accent == "Scottish English"`; exclude text starting with `IGNORE`. Use the supplied utterance audio/text rows in source order. |
| English Dialects, Scottish female and male | [ylacombe/english_dialects](https://huggingface.co/datasets/ylacombe/english_dialects), revision `ed3d69abb765fd304ccdcc646ec6ca3c49740d15`; upstream [OpenSLR SLR83](https://www.openslr.org/83/) Scottish ZIPs and transcript index are also available. | Load `scottish_female/train` (894 rows), then `scottish_male/train` (1,649); concatenate in that order. No additional row filtering or audio segmentation is present in the loader. |

For the Hugging Face sources, install the optional corpus-preparation tool with
`pip install datasets` (not needed for bundled examples), then download/cache:

```python
from datasets import load_dataset
edacc = load_dataset("edinburghcstr/edacc", split="test",
                     revision="d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66")
female = load_dataset("ylacombe/english_dialects", "scottish_female", split="train",
                      revision="ed3d69abb765fd304ccdcc646ec6ca3c49740d15")
male = load_dataset("ylacombe/english_dialects", "scottish_male", split="train",
                    revision="ed3d69abb765fd304ccdcc646ec6ca3c49740d15")
```

These calls obtain upstream rows; apply the table's filters before assigning
indices. The release CLI accepts audio files or a JSON list of files, not corpus names. Keep
full corpora outside the release, e.g. under ignored `data/`, and export the chosen
utterances/references to an audio file list for `transcribe`. The paper's dev/test membership
is frozen in [paper_results/splits.json](../paper_results/splits.json); it differs from
upstream split names and must not be recreated by randomly splitting these rows.

Preparation checked against archived `src/datasets.py`, `src/benchmark.py`,
`scripts/benchmarking/run_whisperx_benchmark.py` and
`scripts/finetuning/{prepare_manifest,chunking}.py` in backup branch
`archive/pre-release-simplification`:

- **Audio:** historical loaders preserve native rates; ASR adapters resample to
  16 kHz. The active [transcriber](../asr_pipeline/transcription.py) decodes each file
  as 16 kHz mono with librosa. No explicit silence trimming, denoising or loudness
  normalization is applied. WhisperX has its own VAD/alignment stage.
- **Segmentation:** source recordings are used as supplied. Active sentence
  confidence splits fused **text** using SaT `sat-3l`, then content-preserving
  boundary rules in [sentence_boundaries.py](../asr_pipeline/sentence_boundaries.py): split missed
  punctuation boundaries except abbreviations/initials, and merge a reporting
  clause ending in comma/colon with a following quotation. This does not cut audio.
  Archived fine-tuning separately resampled to 16 kHz and chunked long samples at
  about 28 seconds using WhisperX-derived reference timings, with a proportional
  fallback. The retired fine-tuned Whisper inference path used fixed 28-second
  windows. Neither path runs in the active four-model quickstart.
- **Text/scoring:** reference files retain original wording. WER/MBR normalization
  in [text_normalization.py](../asr_pipeline/text_normalization.py) removes `<...>` annotation tags, lowercases,
  standardizes curly quotes/dashes, removes punctuation except apostrophes, and
  collapses whitespace. Historical benchmarking also skipped references containing
  `IGNORE_TIME_SEGMENT_IN_SCORING` or only annotation tags. Active evaluation skips
  empty/tag-only references; fresh-clip prediction never filters from reference
  content. Confidence alignment uses its own tokenization, not WER normalization.

**Unknown manual steps:** no preparation script establishes how the downloaded
Common Voice archive became the local TSV/audio layout, or proves any earlier
manual edits/trimming beyond the supplied files. Historical splitting recordings
and Hugging Face repackaging are not recreated by the active CLI. Exact historical
producer invocations and additional undocumented edits remain unknown. The table
states observed loader behavior, not a claim that all earlier preparation is known.
