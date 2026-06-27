"""
run_eval_suite.py

Rule-based evaluation suite. For each model's subset benchmark JSON:
1. Aligns ref/hyp words via Levenshtein (jiwer)
2. Classifies each error (substitution/deletion) by category using:
   - spaCy NER for named entities (with function-word filtering)
   - word lists for negation, pronouns, profanity/informal
   - Scottish glossary for dialect words (with common-word exclusions)
3. Computes error rate per category: errors_in_category / total_words_in_category

Models and datasets are discovered dynamically by scanning the subsets
directory for files matching {model}_{dataset}_sub{N}.json — no hardcoded
model list. Add a new model's subset file and it's picked up automatically.

Usage:
    python scripts/evaluation/eval_suite/run_eval_suite.py --dataset commonvoice
    python scripts/evaluation/eval_suite/run_eval_suite.py --dataset all
    python scripts/evaluation/eval_suite/run_eval_suite.py --sample negation --model qwen3asr --n 5 --mode errors
    python scripts/evaluation/eval_suite/run_eval_suite.py --list   # show discovered models/datasets
"""

import json
import os
import re
import argparse
from collections import defaultdict
import spacy
from jiwer import process_words

SUBSETS_DIR = "results/benchmarks/subsets"
GLOSSARY_PATH = "data/scottish_glossary.json"

# known dataset names — used to split {model}_{dataset}_sub{N}.json filenames
# correctly even when model names themselves contain underscores
KNOWN_DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


def discover_models_and_datasets(subsets_dir=SUBSETS_DIR):
    """
    Scan the subsets directory and infer (model, dataset) pairs from
    filenames matching the pattern: {model}_{dataset}_sub{N}.json

    No hardcoded model list — models are whatever appears in the filenames.
    Dataset names are matched against KNOWN_DATASETS so that model names
    containing underscores (e.g. "qwen3asr") still parse correctly.

    Returns:
        models: sorted list of unique model names found
        datasets: sorted list of unique dataset names found
        files: dict mapping (model, dataset) -> filename
    """
    dataset_pattern = "|".join(re.escape(d) for d in KNOWN_DATASETS)
    pattern = re.compile(rf"^(.+?)_({dataset_pattern})_sub\d+\.json$")

    files = {}
    models = set()
    datasets = set()

    if not os.path.isdir(subsets_dir):
        return [], [], {}

    for fname in sorted(os.listdir(subsets_dir)):
        match = pattern.match(fname)
        if match:
            model, dataset = match.groups()
            files[(model, dataset)] = fname
            models.add(model)
            datasets.add(dataset)

    return sorted(models), sorted(datasets), files


MODELS, DATASETS, SUBSET_FILES = discover_models_and_datasets()

# ── Word lists ───────────────────────────────────────────────────────────────

NEGATION_WORDS = {
    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor",
    "cannot", "can't", "won't", "wouldn't", "shouldn't", "couldn't", "didn't",
    "doesn't", "don't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't",
    # Scottish dialect negations
    "wisnae", "wasnae", "dinnae", "didnae", "cannae", "willnae", "wouldnae",
    "couldnae", "shouldnae", "disnae", "hasnae", "havenae", "isnae", "arenae",
    "nae", "no'",
}

PRONOUNS = {
    "i", "you", "he", "she", "we", "they", "it", "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "our", "their", "its", "mine", "yours", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "ourselves", "themselves", "itself",
}

PROFANITY_INFORMAL = {
    "shit", "shite", "bloody", "bollocks", "crap", "damn", "hell", "bugger",
    "arse", "bastard", "piss", "pissed", "fuck", "fucking", "fucked",
}

# common English function words that occasionally appear in the Scottish
# glossary (secondary Scots senses) but cause false positives when matched
# against ordinary English usage
GLOSSARY_EXCLUSIONS = {
    "an", "as", "but", "by", "for", "i", "in", "no", "on", "or",
    "over", "she", "the", "we",
}

# words that should never be tagged as named entities even if spaCy
# includes them inside an entity span boundary (articles, prepositions,
# conjunctions that get swept into a span like "the Napoleonic Wars")
NER_SPAN_EXCLUSIONS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be",
}

# ── Normalisation ────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r"[^\w\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Categorisation ───────────────────────────────────────────────────────────

CATEGORIES = [
    "named_entity", "negation", "dialect_word",
    "profanity_informal", "pronoun", "general",
]


def load_glossary():
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH) as f:
            data = json.load(f)
        if isinstance(data, dict):
            words = set(k.lower() for k in data.keys())
        elif isinstance(data, list):
            words = set()
            for item in data:
                if isinstance(item, str):
                    words.add(item.lower())
                elif isinstance(item, dict) and "word" in item:
                    words.add(item["word"].lower())
        else:
            words = set()
        # remove common English function words that happen to have
        # secondary Scots senses — they cause false positives on
        # ordinary English usage
        words -= GLOSSARY_EXCLUSIONS
        return words
    return set()


def categorise_word(word: str, ner_labels: dict, glossary: set) -> str:
    """
    Categorise a single (lowercased, normalised) reference word.
    Returns one category from CATEGORIES.
    Priority order matters — check most specific categories first.

    NOTE: Numbers are deliberately not categorised separately — digit vs
    spelled-out number mismatches (e.g. "15" vs "fifteen") create alignment
    noise that isn't a genuine error. This is a known limitation; number
    errors fall into "general".
    """
    word_lower = word.lower()

    if word_lower in ner_labels:
        return "named_entity"
    if word_lower in NEGATION_WORDS:
        return "negation"
    if word_lower in PROFANITY_INFORMAL:
        return "profanity_informal"
    if word_lower in glossary:
        return "dialect_word"
    if word_lower in PRONOUNS:
        return "pronoun"
    return "general"


def get_ner_labels(nlp, text: str) -> dict:
    """
    Run spaCy NER on text, return dict mapping lowercase token text -> True
    for tokens that are part of a named entity span.

    Excludes function words (articles, prepositions, conjunctions) that
    spaCy sometimes includes inside an entity span boundary but are not
    semantically part of the name (e.g. "the" in "the Napoleonic Wars").
    Uses spaCy's own POS tagging to additionally exclude any token tagged
    as DET, ADP, CCONJ, or SCONJ within an entity span.
    """
    doc = nlp(text)
    labels = {}
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP"}:
            for token in ent:
                token_lower = token.text.lower()
                if token_lower in NER_SPAN_EXCLUSIONS:
                    continue
                if token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ"}:
                    continue
                labels[token_lower] = True
    return labels


# ── Alignment ────────────────────────────────────────────────────────────────

NEGATION_DIALECT_PAIRS = {
    # ref_word -> set of hyp words that are equivalent negation forms
    "no":      {"not", "no", "n't"},
    "nae":     {"not", "no", "n't"},
    "wisnae":  {"wasn't", "wasnt", "was"},  # "was" only if "not" elsewhere — approximate
    "didnae":  {"didn't", "didnt"},
    "dinnae":  {"don't", "dont"},
    "cannae":  {"can't", "cant"},
    "willnae": {"won't", "wont"},
    "wouldnae": {"wouldn't", "wouldnt"},
    "couldnae": {"couldn't", "couldnt"},
    "shouldnae": {"shouldn't", "shouldnt"},
    "disnae":  {"doesn't", "doesnt"},
    "hasnae":  {"hasn't", "hasnt"},
    "havenae": {"haven't", "havent"},
    "isnae":   {"isn't", "isnt"},
    "arenae":  {"aren't", "arent"},
}


def is_negation_form_change(ref_word: str, hyp_words: list) -> bool:
    """
    Check if a negation substitution is just a dialect/form change
    (e.g. "no"->"not", "didnae"->"didn't") rather than a real error —
    i.e. the negation is preserved in the hypothesis.
    """
    ref_lower = ref_word.lower()
    hyp_lower_set = {h.lower() for h in hyp_words}

    # any hyp word is itself a recognised negation word
    if hyp_lower_set & NEGATION_WORDS:
        return True

    # check known dialect equivalence pairs
    equivalents = NEGATION_DIALECT_PAIRS.get(ref_lower, set())
    if hyp_lower_set & equivalents:
        return True

    return False


def get_error_ref_words(ref: str, hyp: str):
    """
    Align ref and hyp. Return (ref_words, error_items) where error_items
    is a list of (ref_word, hyp_words_at_position) tuples for substitutions
    and deletions — these are the "ground truth" words the model got wrong.
    """
    ref_norm = normalise(ref)
    hyp_norm = normalise(hyp)

    if not ref_norm or not hyp_norm:
        return [], []

    try:
        output = process_words(ref_norm, hyp_norm)
    except Exception:
        return [], []

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    error_items = []

    for chunk in output.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            r_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
            h_words = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
            for w in r_words:
                error_items.append((w, h_words))

    return ref_words, error_items


def sample_category_errors(model, dataset, category, nlp, glossary, n=5):
    """
    Print n sample errors for a given category to spot-check correctness.
    Shows ref, hyp, and which words were classified into this category.
    """
    path = os.path.join(SUBSETS_DIR, SUBSET_FILES.get((model, dataset), ""))
    if not os.path.exists(path):
        print(f"  No subset file found for model={model}, dataset={dataset}")
        return

    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    shown = 0

    for s in samples:
        if shown >= n:
            break

        ref = s.get("ref", "")
        hyp = s.get("hyp", "")
        if not ref or not hyp:
            continue

        ner_labels = get_ner_labels(nlp, ref)
        ref_words, error_items = get_error_ref_words(ref, hyp)
        if not ref_words:
            continue

        cat_errors = []
        for ref_word, hyp_words in error_items:
            cat = categorise_word(ref_word, ner_labels, glossary)
            if cat != category:
                continue
            if cat == "negation" and is_negation_form_change(ref_word, hyp_words):
                cat_errors.append(f"{ref_word} (form-change, filtered)")
                continue
            cat_errors.append(f"{ref_word} -> {hyp_words}")

        if cat_errors:
            print(f"\n  Sample {shown+1}:")
            print(f"    REF: {ref[:150]}")
            print(f"    HYP: {hyp[:150]}")
            print(f"    {category} items: {cat_errors}")
            shown += 1


def sample_category_total(model, dataset, category, nlp, glossary, n=5):
    """
    Print n samples showing words classified into this category
    (regardless of whether they were errors) — to verify categorisation logic.
    """
    path = os.path.join(SUBSETS_DIR, SUBSET_FILES.get((model, dataset), ""))
    if not os.path.exists(path):
        print(f"  No subset file found for model={model}, dataset={dataset}")
        return

    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    shown = 0

    for s in samples:
        if shown >= n:
            break

        ref = s.get("ref", "")
        hyp = s.get("hyp", "")
        if not ref:
            continue

        ner_labels = get_ner_labels(nlp, ref)
        ref_norm = normalise(ref)
        ref_words = ref_norm.split()

        cat_words = [w for w in ref_words
                     if categorise_word(w, ner_labels, glossary) == category]

        if cat_words:
            print(f"\n  Sample {shown+1}:")
            print(f"    REF: {ref[:150]}")
            print(f"    {category} words found: {cat_words}")
            shown += 1


# ── Main evaluation per model/dataset ───────────────────────────────────────

def run_model_dataset(model, dataset, nlp, glossary):
    path = os.path.join(SUBSETS_DIR, SUBSET_FILES.get((model, dataset), ""))
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])

    category_totals = defaultdict(int)  # total ref words per category
    category_errors = defaultdict(int)  # error ref words per category

    for s in samples:
        ref = s.get("ref", "")
        hyp = s.get("hyp", "")
        if not ref or not hyp:
            continue

        ner_labels = get_ner_labels(nlp, ref)  # NER on original (cased) ref

        ref_words, error_items = get_error_ref_words(ref, hyp)
        if not ref_words:
            continue

        for w in ref_words:
            cat = categorise_word(w, ner_labels, glossary)
            category_totals[cat] += 1

        for ref_word, hyp_words in error_items:
            cat = categorise_word(ref_word, ner_labels, glossary)
            # skip negation dialect/form changes — negation preserved
            if cat == "negation" and is_negation_form_change(ref_word, hyp_words):
                continue
            category_errors[cat] += 1

    profile = {}
    for cat in CATEGORIES:
        total = category_totals.get(cat, 0)
        errors = category_errors.get(cat, 0)
        rate = errors / total if total > 0 else None
        profile[cat] = {"errors": errors, "total": total, "rate": rate}

    return profile


def print_profile(model, dataset, profile):
    print(f"\n── {model} / {dataset} ─────────────────────────────────")
    print(f"  {'Category':<20} {'Errors':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'-'*48}")
    for cat in CATEGORIES:
        p = profile[cat]
        rate_str = f"{p['rate']*100:.1f}%" if p['rate'] is not None else "—"
        print(f"  {cat:<20} {p['errors']:>8} {p['total']:>8} {rate_str:>8}")


def print_rankings(all_profiles, dataset, models):
    print(f"\n{'='*60}")
    print(f"RANKINGS — {dataset} (lower error rate = more reliable)")
    print(f"{'='*60}")

    for cat in CATEGORIES:
        rows = []
        for model in models:
            profile = all_profiles.get((model, dataset))
            if not profile:
                continue
            p = profile[cat]
            if p["total"] >= 5 and p["rate"] is not None:  # need enough samples
                rows.append((model, p["rate"], p["total"]))

        if not rows:
            print(f"  {cat:<20} insufficient data")
            continue

        rows.sort(key=lambda x: x[1])
        ranking_str = " < ".join(f"{m}({r*100:.1f}%, n={n})" for m, r, n in rows)
        print(f"  {cat:<20} {ranking_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--sample", type=str, default=None,
                        choices=CATEGORIES,
                        help="Sample N errors from this category (use with --model and --n)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (must match a discovered subset file). "
                             "Defaults to the first discovered model.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--mode", type=str, default="errors",
                        choices=["errors", "total"],
                        help="'errors' shows error words in category, "
                             "'total' shows all words classified into category")
    parser.add_argument("--list", action="store_true",
                        help="List discovered models and datasets, then exit")
    args = parser.parse_args()

    if args.list:
        print(f"Discovered models:   {MODELS}")
        print(f"Discovered datasets:  {DATASETS}")
        print(f"\nFiles found:")
        for (m, d), fname in sorted(SUBSET_FILES.items()):
            print(f"  ({m}, {d}) -> {fname}")
        return

    if not MODELS:
        print(f"No subset files discovered in {SUBSETS_DIR}. "
              f"Expected filenames like '{{model}}_{{dataset}}_sub150.json'.")
        return

    print("Loading spaCy model and glossary...")
    nlp = spacy.load("en_core_web_sm")
    glossary = load_glossary()
    print(f"Glossary: {len(glossary)} dialect words loaded")
    print(f"Discovered models: {MODELS}")
    print(f"Discovered datasets: {DATASETS}\n")

    if args.sample:
        model = args.model or MODELS[0]
        dataset = args.dataset if args.dataset != "all" else DATASETS[0]
        print(f"Sampling '{args.sample}' ({args.mode}) for {model}/{dataset}:")
        if args.mode == "errors":
            sample_category_errors(model, dataset, args.sample, nlp, glossary, args.n)
        else:
            sample_category_total(model, dataset, args.sample, nlp, glossary, args.n)
        return

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_profiles = {}
    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"No subset files found for dataset='{dataset}'. Skipping.")
            continue
        for model in MODELS:
            profile = run_model_dataset(model, dataset, nlp, glossary)
            if profile:
                all_profiles[(model, dataset)] = profile
                print_profile(model, dataset, profile)

        print_rankings(all_profiles, dataset, MODELS)

    # save results
    output = {
        f"{m}_{d}": p for (m, d), p in all_profiles.items()
    }
    os.makedirs("results/eval_suite", exist_ok=True)
    out_path = "results/eval_suite/error_profiles.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()