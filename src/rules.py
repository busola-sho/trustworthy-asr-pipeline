"""
src/rules.py

Converts eval suite error_profiles.json into auto-generated, model-centred
selector rules — pooling error counts across ALL datasets per model/category.

A rule is only emitted if:
  1. The aggregate (pooled) error rate gap between best and second-best
     model exceeds min_gap_pp (default 1.0pp), AND
  2. The best model wins a majority of decisive (non-tied) per-dataset
     contests among the datasets it has data for.

Genuine ties are excluded from win/loss counts entirely. If ties exist,
a note is appended telling the selector to use contextual judgement —
without referencing dataset names.

Usage:
    from src.rules import build_rules_text
    rules_text = build_rules_text(min_gap_pp=1.0)
"""

import json
import os
from collections import defaultdict

PROFILES_PATH     = "results/eval_suite/error_profiles.json"
RULES_OUTPUT_PATH = "results/eval_suite/selector_rules.txt"

KNOWN_DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]

CATEGORY_LABELS = {
    "named_entity":       "named entities (people's names, place names, organisations)",
    "negation":           "negations",
    "dialect_word":       "Scottish dialect words",
    "profanity_informal": "profanity and informal expressions",
    "pronoun":            "pronouns",
}

CATEGORIES_TO_RULE = ["named_entity", "negation", "dialect_word",
                      "profanity_informal", "pronoun"]

MIN_N_PER_DATASET  = 10
MIN_N_AGGREGATE    = 20
DEFAULT_MIN_GAP_PP = 1.0
MIN_DATASET_MAJORITY = 0.5


def load_profiles(path: str = PROFILES_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def discover_models_and_datasets(profiles: dict):
    models, datasets, pairs = set(), set(), {}
    for key in profiles.keys():
        for ds in KNOWN_DATASETS:
            if key.endswith(f"_{ds}"):
                model = key[: -len(f"_{ds}")]
                models.add(model)
                datasets.add(ds)
                pairs[(model, ds)] = key
                break
    return sorted(models), sorted(datasets), pairs


def pool_category_across_datasets(profiles, models, datasets, category):
    aggregate_errors = defaultdict(int)
    aggregate_totals = defaultdict(int)
    per_dataset      = defaultdict(list)

    for model in models:
        for dataset in datasets:
            p = profiles.get(f"{model}_{dataset}", {}).get(category)
            if not p or p.get("total", 0) == 0:
                continue
            aggregate_errors[model] += p["errors"]
            aggregate_totals[model] += p["total"]
            if p["total"] >= MIN_N_PER_DATASET and p.get("rate") is not None:
                per_dataset[model].append((dataset, p["rate"], p["total"]))

    aggregate = {
        model: (aggregate_errors[model] / aggregate_totals[model], aggregate_totals[model])
        for model in models
        if aggregate_totals.get(model, 0) > 0
    }
    return aggregate, per_dataset


def best_model_wins_majority(best_model, per_dataset, models):
    best_datasets = {d: r for d, r, _ in per_dataset.get(best_model, [])}
    if not best_datasets:
        return False, []

    wins, decisive, tie_datasets = 0, 0, []

    for dataset, best_rate in best_datasets.items():
        tied_models, beaten = [], None
        for other in models:
            if other == best_model:
                continue
            other_rates = {d: r for d, r, _ in per_dataset.get(other, [])}
            if dataset not in other_rates:
                continue
            if other_rates[dataset] < best_rate:
                beaten = other
            elif other_rates[dataset] == best_rate:
                tied_models.append(other)

        if beaten:
            decisive += 1
        elif tied_models:
            tie_datasets.append((dataset, tied_models))
        else:
            decisive += 1
            wins += 1

    if decisive == 0:
        return False, tie_datasets
    return (wins / decisive) >= MIN_DATASET_MAJORITY, tie_datasets


def generate_model_centred_rules(profiles, models, datasets,
                                  min_gap_pp=DEFAULT_MIN_GAP_PP,
                                  show_breakdown=False):
    lines = []

    for category in CATEGORIES_TO_RULE:
        aggregate, per_dataset = pool_category_across_datasets(
            profiles, models, datasets, category)

        if show_breakdown:
            print(f"\n  [{category}]")
            for model in models:
                if model in aggregate:
                    rate, n = aggregate[model]
                    per_ds = ", ".join(
                        f"{d}:{r*100:.1f}%(n={n_})"
                        for d, r, n_ in per_dataset.get(model, [])
                    )
                    print(f"    {model}: pooled={rate*100:.1f}% (n={n})  [{per_ds}]")

        valid = {m: v for m, v in aggregate.items() if v[1] >= MIN_N_AGGREGATE}
        if len(valid) < 2:
            continue

        ranked                          = sorted(valid.items(), key=lambda x: x[1][0])
        best_model,   (best_rate,   best_n)   = ranked[0]
        second_model, (second_rate, second_n) = ranked[1]
        gap_pp = (second_rate - best_rate) * 100

        if gap_pp < min_gap_pp:
            continue

        majority_held, tie_datasets = best_model_wins_majority(
            best_model, per_dataset, models)
        if not majority_held:
            continue

        cat_label = CATEGORY_LABELS[category]
        rule = (
            f"- {cat_label.upper()}: the {best_model} transcript is empirically "
            f"more reliable on {cat_label} ({best_rate*100:.1f}% pooled error rate "
            f"across all datasets, n={best_n}, vs {second_rate*100:.1f}% for the "
            f"next best model). If the {best_model} transcript disagrees with the "
            f"others here, lean toward trusting it."
        )

        if tie_datasets:
            tied_flat = sorted(
                set(m for _, ms in tie_datasets for m in ms) | {best_model}
            )
            rule += (
                f" Note: in some cases {', '.join(tied_flat)} perform "
                f"near-identically on {cat_label} with no clear winner — in "
                f"those cases, do not default to {best_model} automatically; "
                f"use contextual judgement instead."
            )

        lines.append(rule)

    return (
        "\n".join(lines) if lines
        else "(No category had a consistent cross-dataset winner — use GENERAL RULE only.)"
    )


def build_rules_text(min_gap_pp: float = DEFAULT_MIN_GAP_PP, save: bool = True) -> str:
    """
    Main importable entry point. Builds and optionally saves the rules text.

    Args:
        min_gap_pp: minimum pooled error-rate gap (pp) to emit a rule
        save:       whether to write rules to RULES_OUTPUT_PATH

    Returns:
        rules text string, ready to be inserted into a selector prompt
    """
    profiles = load_profiles()
    models, datasets, _ = discover_models_and_datasets(profiles)
    rules_text = generate_model_centred_rules(profiles, models, datasets,
                                              min_gap_pp=min_gap_pp)
    if save:
        os.makedirs(os.path.dirname(RULES_OUTPUT_PATH), exist_ok=True)
        with open(RULES_OUTPUT_PATH, "w") as f:
            f.write(rules_text)

    return rules_text