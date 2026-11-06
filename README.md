# Beyond WER for Scottish speech: ASR, fusion and sentence confidence

This repo brings together Qwen3-ASR, WhisperX, Parakeet and Wav2Vec2 to transcribe
Scottish speech. You can combine their transcripts, estimate sentence confidence,
and evaluate predictions against an optional reference. Your new results go in
`runs/`; the saved paper results stay in `paper_results/`.

## Install

To run the pipeline, you’ll need Python 3.11 or newer and the models described
below. Start in the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m asr_pipeline --help
```

`requirements.txt` installs the project with the ASR and reporting extras from
`pyproject.toml`, including SaT sentence splitting. Model weights are
separate: your first prediction run downloads any missing ASR and SaT weights.
They need substantial disk space and memory. After that, add `--cached-models-only`
to use the downloaded weights without fetching more.

Next, install [Ollama](https://ollama.com/download). If it isn’t already running,
start `ollama serve` in another terminal. Download the two LLMs the pipeline uses:

```bash
ollama pull gemma4:12b
ollama pull phi4:14b
```

Gemma combines transcripts and gives verbalised confidence. Phi-4 evaluates
severity and can help align sentences to source transcripts. The pipeline connects
to Ollama at `http://localhost:11434`; use `--ollama-url` if yours runs elsewhere.

## Run a bundled example

Once everything is installed, try one of the
[five Scottish audio examples](examples/scottish/README.md) included in this repo.
Each has a matching reference TXT file. They come from the Scottish female (three
clips) and male (two clips) subsets of OpenSLR SLR83, so you don’t need a corpus download:

```bash
python -m asr_pipeline run \
  --audio examples/scottish/scm_07508_01418625966.wav \
  --reference-file examples/scottish/scm_07508_01418625966.txt \
  --output-dir runs/scottish-one
```

When the run finishes, you’ll see the reference, ASR transcripts, combination
outputs and sentence confidence in the terminal. Open
`runs/scottish-one/walkthrough.html` to listen to the audio and compare the same
results, with links to the JSON files and logs.

To try all five examples, run them one after another with this loop:

```bash
for audio in examples/scottish/*.wav; do
  name=$(basename "$audio" .wav)
  python -m asr_pipeline run --audio "$audio" \
    --reference-file "${audio%.wav}.txt" --output-dir "runs/scottish-five/$name" || break
done
```

Each example gets its own folder under `runs/scottish-five/`. Choose a new output
folder if you repeat a run; `runs/` is ignored by Git. The example audio and text
use [CC BY-SA 4.0](examples/scottish/LICENSE). That licence applies to the data;
no separate code licence is declared.

## Use your own audio

Have your own recording? Use the same command with your audio file:

```bash
python -m asr_pipeline run --audio my-recording.wav --output-dir runs/my-recording
```

This produces transcripts and confidence signals without reference-based evaluation.
If you also have a reference, save it as a UTF-8 TXT file and add it here:

```bash
python -m asr_pipeline run --audio my-recording.wav \
  --reference-file my-reference.txt --output-dir runs/my-evaluated-recording
```

You can also pass a direct HTTP(S) audio URL to `--audio`. The file is downloaded
once into your run’s `input/` folder. An audio URL still needs network access with
`--cached-models-only`, since that option applies only to model weights.

With a reference, you’ll also get WER and severity scores. The reference is used
**only for evaluation**, never for transcription, fusion or confidence prediction.
Without one, you still get transcripts, sentence splits and confidence signals;
severity fields remain null.

You’ll find these files inside your chosen output folder:

| Folder/file | Contents |
|---|---|
| `asr/` | Four transcripts and word confidences |
| `fusion/` | Selection, anchored fusion, unanchored fusion, ROVER and MBR outputs |
| `evaluation/` | WER and severity, when a reference was supplied |
| `confidence/` | Sentences, verbalised confidence, agreement and raw ASR confidence |
| `walkthrough.html`, `summary.txt` | Readable results |
| `run.json`, `requests.jsonl`, `logs/` | Completed steps, raw LLM responses and ASR logs |

If a step fails, its partial outputs are kept for inspection. Existing run folders
are never replaced. Calibrated probabilities, normalized ASR confidence and learned
proxies need suitable four-model parameters; this command neither fits them nor
borrows the paper’s five-model parameters. Their status is recorded in `run.json`,
without a separate list in the terminal summary. Treat verbalised scores as uncalibrated.
Judge scores can contradict their explanations, and you’ll be able to inspect both.

## Run sentence confidence separately

The `run` command already estimates confidence on unanchored fusion. If you’d like
to calculate agreement separately, first finish the bundled example above, then
use its saved fusion and sentence files:

```bash
python -m asr_pipeline confidence agreement --dataset clip --split full \
  --fusion-file runs/scottish-one/fusion/unanchored.json \
  --sentences-file runs/scottish-one/confidence/segments/sentence_segments_clip_full.json \
  --confidence-dir runs/scottish-agreement --include-unlabelled
```

You’ll get `runs/scottish-agreement/cross_model_agreement/cross_model_agreement_clip_full.json`.
This uses the four saved ASR transcripts, so there’s no LLM call.

For a new verbalised score, keep Ollama running and use those same inputs with
`verbalized --variant confscore`:

```bash
python -m asr_pipeline confidence verbalized --variant confscore --dataset clip --split full \
  --fusion-file runs/scottish-one/fusion/unanchored.json \
  --sentences-file runs/scottish-one/confidence/segments/sentence_segments_clip_full.json \
  --confidence-dir runs/scottish-verbalized --include-unlabelled
```

The scores go in `runs/scottish-verbalized/verbalized_confidence/verbalized_confscore_clip_full.json`.
Here, `--include-unlabelled` lets you score sentences without evaluation labels.
For paper inputs, choose the dataset with `--dataset` and dev, test or Shetland full
with `--split`. Use `--fusion-dir` for files that follow the paper filename pattern.

## Run ASR or fusion separately

If you want to run individual stages, start by saving this list as
`audio_files.json` in the repository folder. Audio paths are relative to that file:

```json
[{"audio":"examples/scottish/scm_07508_01418625966.wav","dataset_index":0,"ref":""}]
```

The command below writes Qwen’s transcript to `runs/separate-asr/qwen.json`.
Repeat it with `whisperx`, `parakeet` and `wav2vec2`, changing the output filename
for each model:

```bash
python -m asr_pipeline transcribe --model qwen --audio-files audio_files.json \
  --dataset clip --split full --output-file runs/separate-asr/qwen.json
```

Once all four transcripts are ready, save their paths in `transcript_files.json`:

```json
{"dataset":"clip","split":"full","sources":{
  "qwen":"runs/separate-asr/qwen.json","whisperx":"runs/separate-asr/whisperx.json",
  "parakeet":"runs/separate-asr/parakeet.json","wav2vec2":"runs/separate-asr/wav2vec2.json"
}}
```

Now combine those transcripts with unanchored fusion:

```bash
python -m asr_pipeline fuse --transcript-files transcript_files.json \
  --method unanchored --output-file runs/separate-fusion.json
```

You’ll find the combined transcript in `runs/separate-fusion.json`. You can also
choose `selection`, `anchored`, `rover` or `mbr` with `--method`. ROVER and MBR need
no LLM; selection returns one source transcript, while anchored fusion starts from
Qwen and follows the fixed correction prompt. Paths are relative to the JSON file
that lists them. Older CLI option spellings still work.

## Regenerate paper tables and figures

To work with the paper’s **saved results**, use the reporting command below.
It makes tables and figures without generating new ASR or LLM predictions.
You’ll first need the files listed in `paper_results/input_files.json`: a fresh
clone doesn’t include these per-item corpus files because they aren’t cleared for release.

```bash
python -m asr_pipeline report \
  --saved-results-dir paper_results --output-dir runs/paper-report \
  --splits-file paper_results/splits.json \
  --input-files paper_results/input_files.json \
  --original-results-dir paper_results --reports tables confidence-figures
```

Your reports go in `runs/paper-report/`. The command checks the input checksums
and verifies that it created every expected output, leaving saved paper files alone.
Reporting may fit small statistical calibrators from saved scores, separately from
prediction. See [paper_results/README.md](paper_results/README.md) for the output
list and extra data needed for downstream plots.

Historical confidence results use five-model fusion. The active prediction code
uses four models. Updating code does not convert the historical results.

## Read the code

To follow a run through the code, start with `main()` in
[run_pipeline.py](asr_pipeline/run_pipeline.py). It calls these steps in order:

1. `load_input` copies/downloads audio and reads an optional reference.
2. `transcribe_audio` calls the four ASRs one at a time. Each gets its own process
   so its model weights can be released before the next model loads.
3. `combine_transcripts` runs the five combination methods.
4. `evaluate_transcripts` uses the reference on separate copies. This keeps it
   out of prediction inputs.
5. `estimate_confidence` scores sentences from unanchored fusion.
6. `save_and_display_results` writes the terminal and HTML summaries.

| File | What it handles |
|---|---|
| [transcription.py](asr_pipeline/transcription.py), [asr_models.py](asr_pipeline/asr_models.py) | Audio decoding, ASR calls and word confidence |
| [fusion.py](asr_pipeline/fusion.py), [rover.py](asr_pipeline/rover.py) | Selection, fusion, MBR and ROVER |
| [evaluation.py](asr_pipeline/evaluation.py) | WER and meaning-based severity |
| [sentence_boundaries.py](asr_pipeline/sentence_boundaries.py) | SaT boundary corrections that retain transcript text |
| [confidence/](asr_pipeline/confidence) | Sentence evaluation, alignment and confidence methods |
| [display_results.py](asr_pipeline/display_results.py) | One set of results for terminal and HTML |
| [files.py](asr_pipeline/files.py) | JSON input/output and checks against replacing saved files |
| [reporting/generate_reports.py](paper_results/reporting/generate_reports.py) | Saved-input checks and paper report generation |

Small file-location classes pass input/output paths between confidence and reporting
functions. They do not change other modules' settings. Parallel LLM calls return in
input order so scores stay attached to the correct sentences. Saved JSON field names
and result folder names are kept for compatibility.

For the data side, see [datasets and preparation](docs/datasets.md): it covers
sources, exact filters, resampling, text normalization and gaps in the historical
record. Earlier work is kept locally in `_archive/pre-release/` and on branch
`archive/pre-release-simplification`; the active code doesn’t import it.

## Check changes without running models

When you make a change, run these checks from the repository folder:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s paper_results/reporting -p 'test_*.py'
```

You’ll see the test results in the terminal. These checks use small synthetic
inputs and the bundled files, without loading model weights or rerunning paper experiments.
