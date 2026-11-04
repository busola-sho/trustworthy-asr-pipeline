"""Input and output paths passed explicitly to saved-result reporting functions."""

from dataclasses import dataclass
from pathlib import Path

PAPER_ROOT = Path(__file__).resolve().parents[1]
COMBINATION = "01_model_combination"
CONFIDENCE = "02_sentence_confidence"


@dataclass(frozen=True)
class PaperReportFiles:
    saved_results_dir: Path = PAPER_ROOT
    output_dir: Path = PAPER_ROOT
    split_assignments: dict | None = None

    @property
    def combination_data(self):
        return self.saved_results_dir / COMBINATION / "data"

    @property
    def baseline_dir(self):
        return self.combination_data / "baselines"

    @property
    def grid_dir(self):
        return self.combination_data / "grid"

    @property
    def combination_tables(self):
        return self.output_dir / COMBINATION / "tables"

    @property
    def confidence_data(self):
        return self.saved_results_dir / CONFIDENCE / "data"

    @property
    def segments_dir(self):
        return self.confidence_data / "segments"

    @property
    def verbalized_dir(self):
        return self.confidence_data / "verbalized"

    @property
    def internal_dir(self):
        return self.confidence_data / "model_internal"

    @property
    def agreement_dir(self):
        return self.confidence_data / "cross_model"

    @property
    def proxy_dir(self):
        return self.confidence_data / "learned_proxy"

    @property
    def police_dir(self):
        return self.confidence_data / "police_scotland"

    @property
    def platt_params(self):
        return self.internal_dir / "model_internal_platt_params.json"

    @property
    def confidence_figures(self):
        return self.output_dir / CONFIDENCE / "figures"

    @property
    def confidence_tables(self):
        return self.output_dir / CONFIDENCE / "tables"

    @property
    def calibration_dir(self):
        return self.output_dir / CONFIDENCE / "calibration"
