"""Run one audio file through ASR, fusion, evaluation and sentence confidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

from .config import SOURCE_MODELS
from .fusion import METHODS
from .files import read_json, check_new_output_path, write_json
from .runtime import temporary_environment


class LoggedModelClient:
    """Keep the service response verbatim, including contradictory judgements."""

    def __init__(self, client, path):
        self.client = client
        self.path = path

    def chat(self, **request):
        response = self.client.chat(**request)
        record = {"request": request, "response": response.model_dump()}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return response


def parse_arguments(argv):
    parser = argparse.ArgumentParser(prog="asr-pipeline run", description=__doc__)
    parser.add_argument(
        "--audio", required=True, help="local audio path or direct HTTP(S) audio URL"
    )
    parser.add_argument(
        "--reference-file",
        "--reference",
        type=Path,
        help="optional UTF-8 reference TXT file, used only for evaluation",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--ollama-url",
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="address of the Ollama service",
    )
    parser.add_argument(
        "--cached-models-only",
        "--offline",
        action="store_true",
        help="require cached Hugging Face weights",
    )
    return parser.parse_args(argv)


def load_input(audio_source, reference_file, output_dir):
    """Copy or download audio. Read the optional reference TXT file.

    Keep reference text out of the audio file list passed to transcription."""
    output_dir = check_new_output_path(output_dir)
    reference = None
    if reference_file is not None:
        reference = reference_file.read_text(encoding="utf-8-sig").strip()
        if not reference:
            raise ValueError("Reference TXT file is empty")

    source_url = urlparse(str(audio_source))
    is_url = source_url.scheme in ("http", "https")
    if source_url.scheme and not is_url:
        raise ValueError("Audio URL must use HTTP or HTTPS")
    source = None if is_url else Path(audio_source).resolve(strict=True)
    if source is not None and not source.is_file():
        raise ValueError("Audio input must be a file")

    # Use a fixed basename so downloaded names cannot replace the JSON file lists.
    suffix = Path(source_url.path if is_url else source.name).suffix or ".audio"
    output_dir.mkdir(parents=True)
    input_dir = output_dir / "input"
    input_dir.mkdir()
    (output_dir / "logs").mkdir()
    audio_path = input_dir / f"clip{suffix}"
    if is_url:
        with urlopen(str(audio_source), timeout=60) as response:
            with audio_path.open("xb") as destination:
                shutil.copyfileobj(response, destination)
    else:
        shutil.copyfile(source, audio_path)
    write_json(
        input_dir / "audio.json",
        [
            {
                "audio": audio_path.name,
                "dataset_index": 0,
                "ref": "",  # Reference text is not passed to ASR.
            }
        ],
    )
    write_json(input_dir / "source.json", {"audio_source": str(audio_source)})
    return audio_path, reference


def connect_to_ollama(host):
    from ollama import Client

    service = Client(host=host)
    available_models = {model.model for model in service.list().models}
    missing_models = {"gemma4:12b", "phi4:14b"} - available_models
    if missing_models:
        raise RuntimeError(
            f"Install the missing Ollama models explicitly: {sorted(missing_models)}"
        )
    return service


def transcribe_audio(output_dir, completed_steps):
    """Transcribe the clip with each of the four ASR models.

    Use a separate process for each model so its weights are released afterward."""
    for model in SOURCE_MODELS:
        print(f"Transcribing: {model}", flush=True)
        command = [
            sys.executable,
            "-m",
            "asr_pipeline",
            "transcribe",
            "--model",
            model,
            "--audio-files",
            str(output_dir / "input/audio.json"),
            "--dataset",
            "clip",
            "--split",
            "full",
            "--output-file",
            str(output_dir / "asr" / f"{model}.json"),
        ]
        with (output_dir / "logs" / f"{model}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        completed_steps.append(f"asr/{model}")

    transcript_files = output_dir / "input/sources.json"
    write_json(
        transcript_files,
        {
            "dataset": "clip",
            "split": "full",
            "sources": {model: f"../asr/{model}.json" for model in SOURCE_MODELS},
        },
    )
    return transcript_files


def combine_transcripts(transcript_files, output_dir, client, completed_steps):
    from .fusion import combine_transcript_files

    combined_files = {}
    for method in METHODS:
        print(f"Combining: {method}", flush=True)
        output = output_dir / "fusion" / f"{method}.json"
        combine_transcript_files(transcript_files, output, method, client=client)
        combined_files[method] = output
        completed_steps.append(f"fusion/{method}")
    return combined_files


def segment_transcripts(samples, sentence_splitter, fusion_file):
    """Segment hypothesis text only; evaluation labels are initially absent."""
    from .sentence_boundaries import segment_hypothesis

    transcripts = []
    for sample in samples:
        segments = segment_hypothesis(sample["hyp"], sentence_splitter)
        transcripts.append(
            {
                "dataset_index": sample["dataset_index"],
                "ref": "",
                "hyp": sample["hyp"],
                "raw_alignment_response": None,
                "segments": [
                    {
                        "segment": text.strip(),
                        "corresponding_span": None,
                        "severity": None,
                        "justification": None,
                        "flagged": None,
                    }
                    for text in segments
                    if text.strip()
                ],
            }
        )
    segment_count = sum(len(transcript["segments"]) for transcript in transcripts)
    if not segment_count:
        raise RuntimeError("No sentence segments were produced")
    return {
        "dataset": "clip",
        "split": "full",
        "judge": None,
        "fusion_input": str(fusion_file),
        "asr_models": list(SOURCE_MODELS),
        "architecture": "segmentation only; no reference-based judging",
        "flag_threshold": 2,
        "flag_convention": "y=1 for meaning-altering/flagged (severity>=2), y=0 for meaning-preserved (severity<2)",
        "num_transcripts": len(transcripts),
        "num_segments": segment_count,
        "num_segments_parsed": 0,
        "num_segments_flagged": 0,
        "num_segments_alignment_failed": 0,
        "transcripts": transcripts,
    }


def evaluate_transcripts(
    combined_files, reference, output_dir, client, completed_steps
):
    """The reference enters only copies written to the evaluation directory."""
    if reference is None:
        return {}
    from .evaluation import evaluate_transcript_file

    evaluated_files = {}
    for method, combined_file in combined_files.items():
        print(f"Evaluating: {method}", flush=True)
        evaluation_input = output_dir / "evaluation/inputs" / f"{method}.json"
        data = read_json(combined_file)
        for sample in data["samples"]:
            sample["ref"] = reference
        write_json(evaluation_input, data)
        output = output_dir / "evaluation" / f"{method}.json"
        evaluate_transcript_file(evaluation_input, output, client)
        evaluated_files[method] = output
        completed_steps.append(f"evaluation/{method}")

    from wtpsplit import SaT
    from .confidence.evaluate_sentences import run_dataset as judge_sentences

    fusion_file = combined_files["unanchored"]
    samples = read_json(fusion_file)["samples"]
    evaluation_samples = [{**sample, "ref": reference} for sample in samples]
    judge_sentences(
        "clip",
        client,
        SaT("sat-3l"),
        split="full",
        max_workers=1,
        chunk_size=1,
        samples=evaluation_samples,
        output_dir=output_dir / "evaluation/segments",
        fusion_file=fusion_file,
    )
    completed_steps.append("evaluation/sentences")
    return evaluated_files


def estimate_confidence(fusion_file, output_dir, client, completed_steps):
    """Predict from fixed sentences and four ASR sources, never a reference."""
    from wtpsplit import SaT
    from .confidence.inputs import load_fusion_samples
    from .confidence.verbalized_confidence import run_dataset as verbalized_confidence
    from .confidence.cross_model_agreement import run_dataset as cross_model_agreement
    from .confidence.asr_confidence import (
        compute_raw_segment_scores,
        load_model_word_confidences,
    )

    print("Estimating sentence confidence", flush=True)
    samples = load_fusion_samples("clip", "full", fusion_file=fusion_file)
    segments = segment_transcripts(samples, SaT("sat-3l"), fusion_file)
    write_json(
        output_dir / "confidence/segments/sentence_segments_clip_full.json", segments
    )
    source_hypotheses = {
        sample["dataset_index"]: sample["source_hyps"] for sample in samples
    }
    scores = verbalized_confidence(
        "clip",
        "confscore",
        client,
        split="full",
        max_workers=1,
        chunk_size=1,
        segments_data=segments,
        source_hypotheses=source_hypotheses,
        fusion_file=fusion_file,
        output_dir=output_dir / "confidence/verbalized_confidence",
        include_unlabelled=True,
    )
    cross_model_agreement(
        "clip",
        split="full",
        segments_data=segments,
        source_hypotheses=source_hypotheses,
        output_dir=output_dir / "confidence/cross_model_agreement",
        include_unlabelled=True,
    )
    native_confidences = {
        model: load_model_word_confidences(
            "clip", model, "full", fusion_file=fusion_file
        )
        for model in SOURCE_MODELS
    }
    write_json(
        output_dir / "confidence/model_internal_raw.json",
        {
            "method": "raw_native_word_confidence_aggregation",
            "normalization": "not applied; no parameters fitted or borrowed",
            "asr_models": list(SOURCE_MODELS),
            "transcripts": compute_raw_segment_scores(
                "clip",
                "full",
                native_confidences,
                segments_data=segments,
                include_unlabelled=True,
            ),
        },
    )
    if scores["num_segments_failed"]:
        raise RuntimeError(
            "Incomplete sentence confidence; inspect the saved raw responses"
        )
    completed_steps.append("sentence_confidence_raw_verbalized_agreement")


def save_and_display_results(
    output_dir, audio_path, combined_files, evaluated_files, run_summary, reference=None
):
    from .display_results import (
        build_results_summary,
        render_terminal_summary,
        render_html_summary,
    )

    results = build_results_summary(
        output_dir, combined_files, evaluated_files, reference
    )
    run_summary.update(status="complete", **results)
    terminal_output = render_terminal_summary(results, run_summary["unavailable"])
    print(terminal_output, end="", flush=True)
    (output_dir / "summary.txt").write_text(terminal_output, encoding="utf-8")
    page = render_html_summary(results, run_summary, audio_path, output_dir)
    (output_dir / "walkthrough.html").write_text(page, encoding="utf-8")


def new_run_summary(argv):
    return {
        "command": [sys.executable, "-m", "asr_pipeline", "run", *argv],
        "status": "running",
        "completed": [],
        "unavailable": {
            "normalized_internal_confidence": "requires suitable four-model development normalization parameters",
            "calibrated_confidence": "requires a suitable four-model calibrator; none supplied or fitted",
            "proxy_predictions": "requires a suitable four-model trained proxy; none supplied or fitted",
        },
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(arguments)
    output_dir = check_new_output_path(args.output_dir)
    model_environment = {}
    if args.cached_models_only:
        model_environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    run_summary = new_run_summary(arguments)
    try:
        with temporary_environment(model_environment):
            audio_path, reference = load_input(
                args.audio, args.reference_file, output_dir
            )
            client = LoggedModelClient(
                connect_to_ollama(args.ollama_url), output_dir / "requests.jsonl"
            )
            transcript_files = transcribe_audio(output_dir, run_summary["completed"])
            combined_files = combine_transcripts(
                transcript_files, output_dir, client, run_summary["completed"]
            )
            evaluated_files = evaluate_transcripts(
                combined_files, reference, output_dir, client, run_summary["completed"]
            )
            estimate_confidence(
                combined_files["unanchored"],
                output_dir,
                client,
                run_summary["completed"],
            )
            save_and_display_results(
                output_dir,
                audio_path,
                combined_files,
                evaluated_files,
                run_summary,
                reference,
            )
    except Exception as error:
        run_summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if output_dir.is_dir():
            write_json(output_dir / "run.json", run_summary)
        print(f"Run status: {run_summary['status']}; inspect {output_dir}", flush=True)
