"""Input and output file locations for sentence confidence."""

from dataclasses import dataclass
from pathlib import Path

from asr_pipeline.config import PROJECT_ROOT


@dataclass
class SentenceConfidenceFiles:
    confidence_dir: Path = PROJECT_ROOT / "runs/confidence_four_model"
    paper_results_dir: Path = PROJECT_ROOT / "paper_results"
    fusion_dir: Path | None = None
    fusion_file: Path | None = None
    ollama_url: str = "http://localhost:11434"
    sentences_file: Path | None = None

    def __post_init__(self):
        self.confidence_dir = Path(self.confidence_dir).resolve()
        self.paper_results_dir = Path(self.paper_results_dir).resolve()
        if self.fusion_dir is not None:
            self.fusion_dir = Path(self.fusion_dir).resolve()
        if self.fusion_file is not None:
            self.fusion_file = Path(self.fusion_file).resolve()
        if self.sentences_file is not None:
            self.sentences_file = Path(self.sentences_file).resolve()
        for protected in (
            self.paper_results_dir,
            PROJECT_ROOT / "_archive",
            PROJECT_ROOT / ".git",
        ):
            if (
                self.confidence_dir == protected
                or self.confidence_dir.is_relative_to(protected)
                or protected.is_relative_to(self.confidence_dir)
            ):
                raise ValueError(
                    "Choose a confidence output folder outside saved inputs"
                )

    @property
    def segments_dir(self):
        return self.confidence_dir / "segments"

    @property
    def verbalized_dir(self):
        return self.confidence_dir / "verbalized_confidence"

    @property
    def internal_dir(self):
        return self.confidence_dir / "model_internal_confidence"

    @property
    def agreement_dir(self):
        return self.confidence_dir / "cross_model_agreement"

    @property
    def proxy_dir(self):
        return self.confidence_dir / "learned_proxy"

    @property
    def calibration_dir(self):
        return self.confidence_dir / "calibration"

    @property
    def development_statistics_file(self):
        return self.internal_dir / "norm_stats_pooled.json"

    @property
    def baseline_dir(self):
        return self.paper_results_dir / "01_model_combination/data/baselines"

    @property
    def shetland_dir(self):
        return self.paper_results_dir / "01_model_combination/data/shetland"

    def fusion_path(self, dataset, split):
        if dataset == "shetland" and split != "full":
            raise ValueError("Shetland confidence requires split='full'")
        if self.fusion_file is not None:
            return self.fusion_file
        if self.fusion_dir is not None:
            return (
                self.fusion_dir
                / f"unanchored_fusion_naive_{dataset}_gemma4_{split}.json"
            )
        if dataset == "shetland":
            return self.shetland_dir / "naive_shetland_gemma4sel_full.json"
        if dataset not in ("commonvoice", "edacc", "english_dialects") or split not in (
            "dev",
            "test",
        ):
            raise ValueError(f"No main four-model fusion input for {dataset}/{split}")
        return (
            self.paper_results_dir
            / "01_model_combination/data/grid/unanchored_fusion_naive"
            / f"unanchored_fusion_naive_{dataset}_gemma4_{split}.json"
        )
