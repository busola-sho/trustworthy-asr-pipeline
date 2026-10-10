"""
plot_police_scotland_real_data.py

Visualizes real Police Scotland deployment data returned via their
feedback on the "naive" (4-model fusion) pipeline shared earlier -
109 real recordings, redacted/confidentiality-only output per Sprint 0
privacy requirements (no raw transcripts/audio ever seen directly).

This is genuinely different from every other severity chart tonight:
real deployment-context audio (phone/radio quality, overlapping
speakers), not an academic benchmark dataset. Findings are notably
worse than any benchmark result: mean severity 2.165 (vs <1.9 on the
weakest academic baseline all night), 0% at severity 0, 87% flag rate.

Same visual design as the other severity-distribution figures
(SEVERITY_STACK_COLOURS, enlarged fonts) for direct visual consistency
when placed alongside them.

Usage:
    python plot_police_scotland_real_data.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "writeup_results/figures"

# From the actual Police Scotland feedback JSON (naive pipeline, 109 samples)
SEVERITY_DISTRIBUTION_COUNTS = {0: 0, 1: 14, 2: 63, 3: 32, 4: 0}
N_TOTAL = 109
MEAN_WER = 0.3587
MEAN_SEVERITY = 2.165
FLAG_RATE = 0.8716

SEVERITY_STACK_COLOURS = ["#a8e6a3", "#4caf50", "#f4b942", "#e67e22", "#c0392b"]
SEVERITY_STACK_LABELS = ["0 - no change", "1 - trivial", "2 - ambiguous", "3 - factual", "4 - critical"]

FONT_TITLE = 22
FONT_SUPTITLE = 24
FONT_AXIS_LABEL = 18
FONT_TICK_LABEL = 16
FONT_LEGEND = 18
FONT_BAR_LABEL = 15


def round_percentages_to_100(values):
    floors = [int(v) for v in values]
    remainders = [v - f for v, f in zip(values, floors)]
    deficit = round(100 - sum(floors))
    order = sorted(range(len(values)), key=lambda i: remainders[i], reverse=True)
    result = floors[:]
    for i in order[:max(deficit, 0)]:
        result[i] += 1
    return result


def plot_police_scotland_severity():
    dist_pct = [SEVERITY_DISTRIBUTION_COUNTS[i] / N_TOTAL * 100 for i in range(5)]
    rounded = round_percentages_to_100(dist_pct)

    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.array([0])
    bottom = 0
    for level in range(5):
        value = dist_pct[level]
        ax.bar(x, [value], bottom=bottom, color=SEVERITY_STACK_COLOURS[level],
               edgecolor="black", linewidth=0.8, width=0.5,
               label=SEVERITY_STACK_LABELS[level])
        if value >= 3:
            ax.text(0, bottom + value / 2, f"{rounded[level]}%\n(n={SEVERITY_DISTRIBUTION_COUNTS[level]})",
                    ha="center", va="center", fontsize=FONT_BAR_LABEL,
                    color="black" if level < 3 else "white", fontweight="bold")
        bottom += value

    ax.set_xticks([0])
    ax.set_xticklabels(["Real Police Scotland audio\n(naive pipeline, N=109)"], fontsize=FONT_TICK_LABEL)
    ax.set_ylabel("% of samples", fontsize=FONT_AXIS_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_TICK_LABEL)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.5, 0.5)
    ax.grid(axis="y", alpha=0.3)

    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=FONT_LEGEND,
             frameon=True, markerscale=2.0, title="Severity", title_fontsize=FONT_LEGEND + 1)

    stats_text = (f"Mean severity: {MEAN_SEVERITY:.3f}\n"
                  f"Mean WER: {MEAN_WER*100:.1f}%\n"
                  f"Flag rate: {FLAG_RATE*100:.1f}%")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=FONT_BAR_LABEL,
            ha="right", va="bottom", bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

    fig.suptitle("Real Deployment Data: Police Scotland Feedback\n"
                 "(naive 4-model fusion, pre-fine-tuning pipeline)",
                 fontsize=FONT_SUPTITLE, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.78, 0.94])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "police_scotland_real_data_severity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_police_scotland_severity()
