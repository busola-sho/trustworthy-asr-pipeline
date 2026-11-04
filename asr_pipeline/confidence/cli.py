"""Choose a sentence-confidence stage and pass its input and output paths."""

import argparse
import os
from pathlib import Path

from asr_pipeline.config import PROJECT_ROOT
from .paths import SentenceConfidenceFiles

STAGES = (
    "evaluate-sentences",
    "segment",
    "verbalized",
    "asr-confidence",
    "internal",
    "agreement",
    "proxy",
    "calibrate",
    "plot",
    "reliability",
)


def run_stage(stage, arguments, paths):
    # Import model code only for the selected step, so help needs no model packages.
    if stage in ("evaluate-sentences", "segment"):
        from .evaluate_sentences import main
    elif stage == "verbalized":
        from .verbalized_confidence import main
    elif stage in ("asr-confidence", "internal"):
        from .asr_confidence import main
    elif stage == "agreement":
        from .cross_model_agreement import main
    elif stage == "proxy":
        from .learned_proxy_confidence import main
    elif stage == "calibrate":
        from .calibrate import main
    elif stage == "plot":
        from .plot_confidence_by_severity import main
    else:
        from .plot_reliability_diagrams import main
    return main(arguments, paths=paths)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m asr_pipeline confidence")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--confidence-dir",
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/confidence_four_model",
    )
    parser.add_argument(
        "--paper-results-dir",
        "--paper-root",
        type=Path,
        default=PROJECT_ROOT / "paper_results",
    )
    parser.add_argument(
        "--fusion-dir",
        "--fusion-root",
        type=Path,
        help="folder containing four-model fusion files",
    )
    parser.add_argument(
        "--fusion-file", type=Path, help="one four-model fusion JSON file"
    )
    parser.add_argument(
        "--sentences-file",
        type=Path,
        help="sentence JSON file previously written by run or segment",
    )
    parser.add_argument(
        "--ollama-url",
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    )
    args, stage_arguments = parser.parse_known_args(argv)
    paths = SentenceConfidenceFiles(
        confidence_dir=args.confidence_dir,
        paper_results_dir=args.paper_results_dir,
        fusion_dir=args.fusion_dir,
        fusion_file=args.fusion_file,
        ollama_url=args.ollama_url,
        sentences_file=args.sentences_file,
    )
    return run_stage(args.stage, stage_arguments, paths)
