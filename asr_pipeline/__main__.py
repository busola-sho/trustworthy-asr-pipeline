"""Read command-line options and call the selected pipeline step."""

import argparse
import os
from pathlib import Path
import sys

from .config import SOURCE_MODELS


def report_main(argv):
    from paper_results.reporting.generate_reports import main

    return main(argv)


def create_parser():
    parser = argparse.ArgumentParser(prog="python -m asr_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "run",
        help="one audio clip through all four ASRs, combination and sentence confidence",
    )
    transcribe = commands.add_parser(
        "transcribe", help="transcribe the audio files listed in a JSON file"
    )
    transcribe.add_argument("--model", choices=SOURCE_MODELS, required=True)
    transcribe.add_argument(
        "--audio-files",
        "--audio-manifest",
        type=Path,
        required=True,
        help="JSON list of audio paths and sample indices",
    )
    transcribe.add_argument(
        "--dataset", required=True, help="dataset name to write in the output JSON"
    )
    transcribe.add_argument("--split", default="full", choices=("dev", "test", "full"))
    transcribe.add_argument("--output-file", "--output", type=Path, required=True)
    fusion = commands.add_parser(
        "fuse", help="selection, anchored/unanchored fusion, ROVER or MBR"
    )
    fusion.add_argument(
        "--method",
        choices=("selection", "anchored", "unanchored", "rover", "mbr"),
        required=True,
    )
    fusion.add_argument(
        "--transcript-files",
        "--inputs",
        type=Path,
        required=True,
        help="JSON file naming the four ASR transcript files",
    )
    fusion.add_argument("--output-file", "--output", type=Path, required=True)
    fusion.add_argument(
        "--splits-file",
        type=Path,
        help="optional JSON file containing the paper split indices",
    )
    fusion.add_argument("--model", default="gemma4:12b")
    fusion.add_argument(
        "--ollama-url",
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    evaluate = commands.add_parser(
        "evaluate", help="compare transcripts with references and save WER and severity"
    )
    evaluate.add_argument("--transcript-file", "--input", type=Path, required=True)
    evaluate.add_argument("--output-file", "--output", type=Path, required=True)
    evaluate.add_argument("--model", default="phi4:14b")
    evaluate.add_argument(
        "--ollama-url",
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    commands.add_parser("confidence", help="four-model sentence-confidence stages")
    commands.add_parser(
        "report", help="make tables and figures from saved paper results"
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "confidence":
        from .confidence.cli import main as confidence_main

        return confidence_main(argv[1:])
    if argv and argv[0] == "report":
        return report_main(argv[1:])
    if argv and argv[0] == "run":
        from .run_pipeline import main as run_main

        return run_main(argv[1:])
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        return transcribe_audio_files(args)
    if args.command == "fuse":
        return combine_transcript_files(args)
    return evaluate_file(args)


def transcribe_audio_files(args):
    from .transcription import transcribe_audio_files as run

    return run(args.model, args.audio_files, args.output_file, args.dataset, args.split)


def combine_transcript_files(args):
    from .fusion import combine_transcript_files as run
    from .files import read_transcript_files, check_new_output_path

    check_new_output_path(args.output_file)
    read_transcript_files(args.transcript_files)
    client = None
    if args.method in ("selection", "anchored", "unanchored"):
        from ollama import Client

        client = Client(host=args.ollama_url)
    return run(
        args.transcript_files,
        args.output_file,
        args.method,
        client=client,
        model=args.model,
        splits_file=args.splits_file,
    )


def evaluate_file(args):
    from .files import read_json, check_new_output_path
    from .evaluation import evaluate_transcript_file as run
    from ollama import Client

    check_new_output_path(args.output_file)
    read_json(args.transcript_file)
    return run(
        args.transcript_file, args.output_file, Client(host=args.ollama_url), args.model
    )


if __name__ == "__main__":
    main()
