"""
paired_bootstrap_naive_vs_v2.py

Paired bootstrap confidence interval on the difference in mean severity
between unanchored_fusion_context_v2 and unanchored_fusion_naive, on
the dev split, across all 3 datasets - testing whether the observed
gap (1.129 vs ~1.147) is a statistically reliable difference or within
noise.

METHODOLOGY: matches the EXACT aggregation used for the real headline
numbers - macro-average (mean of 3 per-dataset means), not a simple
pooled average across all samples combined (which would let English
Dialects' large N dominate). So each bootstrap iteration:
  1. For each dataset, resample (with replacement, same size as
     original) from the PAIRED per-sample differences (v2 - naive),
     matched by dataset_index (only samples scored in BOTH conditions).
  2. Compute that resampled dataset's mean difference.
  3. Macro-average the 3 resampled dataset means.
Repeated N_BOOTSTRAP times to build a distribution of the macro-averaged
difference, from which a 95% CI is read off directly (percentile method).

If the CI excludes 0, the difference is statistically reliable at that
level. If it includes 0, the two conditions are not reliably
distinguishable given this data.

Usage:
    python paired_bootstrap_naive_vs_v2.py
"""

import json
import random

DATASETS = ["commonvoice", "edacc", "english_dialects"]
N_BOOTSTRAP = 10000
SEED = 42

PATH_TEMPLATE = "writeup_results/clean_grid/{technique}/{technique}_{dataset}_gemma4_dev.json"


def load_severities_by_index(technique, dataset):
    path = PATH_TEMPLATE.format(technique=technique, dataset=dataset)
    data = json.load(open(path))
    samples = data.get("samples", [])
    return {s["dataset_index"]: s["severity"] for s in samples
            if s.get("dataset_index") is not None and s.get("severity") is not None}


def get_paired_differences(dataset):
    """Returns a list of (v2_severity - naive_severity) for every
    sample scored in BOTH conditions for this dataset."""
    naive = load_severities_by_index("unanchored_fusion_naive", dataset)
    v2 = load_severities_by_index("unanchored_fusion_context_v2", dataset)

    common_indices = set(naive.keys()) & set(v2.keys())
    diffs = [v2[i] - naive[i] for i in common_indices]
    return diffs, len(common_indices), len(naive), len(v2)


def bootstrap_macro_diff(per_dataset_diffs, rng):
    """One bootstrap iteration: resample each dataset's diffs
    independently, take each dataset's mean, macro-average the 3."""
    dataset_means = []
    for diffs in per_dataset_diffs:
        n = len(diffs)
        resampled = [diffs[rng.randrange(n)] for _ in range(n)]
        dataset_means.append(sum(resampled) / n)
    return sum(dataset_means) / len(dataset_means)


def main():
    print("=== Paired per-sample differences (context_v2 - naive), dev split ===\n")

    per_dataset_diffs = []
    observed_dataset_means = []

    for dataset in DATASETS:
        diffs, n_common, n_naive, n_v2 = get_paired_differences(dataset)
        mean_diff = sum(diffs) / len(diffs)
        per_dataset_diffs.append(diffs)
        observed_dataset_means.append(mean_diff)
        print(f"{dataset}: n_common={n_common} (naive_n={n_naive}, v2_n={n_v2})  "
              f"mean_diff={mean_diff:+.4f}")

    observed_macro_diff = sum(observed_dataset_means) / len(observed_dataset_means)
    print(f"\nObserved macro-averaged difference (v2 - naive): {observed_macro_diff:+.4f}")
    print(f"(Negative means v2 has LOWER severity, i.e. v2 is BETTER)\n")

    print(f"Running {N_BOOTSTRAP} bootstrap iterations...")
    rng = random.Random(SEED)
    bootstrap_diffs = [bootstrap_macro_diff(per_dataset_diffs, rng) for _ in range(N_BOOTSTRAP)]
    bootstrap_diffs.sort()

    lower_idx = int(0.025 * N_BOOTSTRAP)
    upper_idx = int(0.975 * N_BOOTSTRAP)
    ci_lower = bootstrap_diffs[lower_idx]
    ci_upper = bootstrap_diffs[upper_idx]

    print(f"\n95% CI on macro-averaged difference (v2 - naive): [{ci_lower:+.4f}, {ci_upper:+.4f}]")

    if ci_lower < 0 < ci_upper:
        print("\n=> CI INCLUDES 0: the difference is NOT statistically reliable.")
        print("   context_v2 and naive are not reliably distinguishable on this data.")
    elif ci_upper < 0:
        print("\n=> CI EXCLUDES 0 (entirely negative): context_v2 is RELIABLY better than naive.")
    else:
        print("\n=> CI EXCLUDES 0 (entirely positive): naive is RELIABLY better than context_v2.")

    # what fraction of bootstrap iterations favoured v2 at all
    frac_v2_better = sum(1 for d in bootstrap_diffs if d < 0) / N_BOOTSTRAP
    print(f"\nFraction of bootstrap iterations where v2 had lower severity than naive: {frac_v2_better*100:.1f}%")


if __name__ == "__main__":
    main()
