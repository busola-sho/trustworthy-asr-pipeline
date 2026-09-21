"""
run_eval_suite.py

Rule-based evaluation suite. For each model's subset benchmark JSON:
1. Aligns ref/hyp words via Levenshtein (jiwer)
2. Classifies each reference word by category:
   - named entities
   - negations
   - Scottish dialect words
   - profanity/informal words
   - pronouns
   - general
3. Computes category error rate:
   errors_in_category / total_words_in_category

Important:
- Empty hypotheses now count as all reference words deleted.
- Negation dialect/form changes are filtered only when negation is preserved.
- NER tagging is done by token position in the normalised reference text,
  not only by token string.
"""

import json
import os
import re
import argparse
from collections import defaultdict

import spacy
from jiwer import process_words
from src.splits import get_indices_for_split


SUBSETS_DIR = "results/benchmarks/subsets"
GLOSSARY_PATH = "data/scottish_glossary.json"

KNOWN_DATASETS = ["commonvoice", "edacc", "english_dialects", "shetland"]


# ── Discovery ────────────────────────────────────────────────────────────────

def discover_models_and_datasets(subsets_dir=SUBSETS_DIR):
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
    "cannot", "can't", "cant", "won't", "wont", "wouldn't", "wouldnt",
    "shouldn't", "shouldnt", "couldn't", "couldnt", "didn't", "didnt",
    "doesn't", "doesnt", "don't", "dont", "isn't", "isnt", "aren't", "arent",
    "wasn't", "wasnt", "weren't", "werent", "hasn't", "hasnt", "haven't", "havent",
    "hadn't", "hadnt",
    "wisnae", "wasnae", "dinnae", "didnae", "cannae", "willnae", "wouldnae",
    "couldnae", "shouldnae", "disnae", "hasnae", "havenae", "isnae", "arenae",
    "nae", "no'",
}

PRONOUNS = {
    "i", "you", "he", "she", "we", "they", "it", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "its", "mine", "yours", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "ourselves", "themselves", "itself",
}

PROFANITY_INFORMAL = {
    "shit", "shite", "bloody", "bollocks", "crap", "damn", "hell", "bugger",
    "arse", "bastard", "piss", "pissed", "fuck", "fucking", "fucked",
}

GLOSSARY_EXCLUSIONS = {
    "an", "as", "but", "by", "for", "i", "in", "no", "on", "or",
    "over", "she", "the", "we",
}

NER_SPAN_EXCLUSIONS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be",
}

NER_ENTITY_LABELS = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP"}

CATEGORIES = [
    "named_entity",
    "negation",
    "dialect_word",
    "profanity_informal",
    "pronoun",
    "general",
]


# ── Normalisation ────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"[^\w\s']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_token(text: str) -> str:
    return normalise(text)


# ── Glossary ─────────────────────────────────────────────────────────────────

def load_glossary():
    if not os.path.exists(GLOSSARY_PATH):
        return set()

    with open(GLOSSARY_PATH) as f:
        data = json.load(f)

    words = set()

    if isinstance(data, dict):
        words = {str(k).lower() for k in data.keys()}
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                words.add(item.lower())
            elif isinstance(item, dict) and "word" in item:
                words.add(str(item["word"]).lower())

    words = {normalise_token(w) for w in words if normalise_token(w)}
    words -= GLOSSARY_EXCLUSIONS
    return words


# ── NER ──────────────────────────────────────────────────────────────────────

def get_ner_word_indices(nlp, text: str) -> set:
    """
    Return indices of normalised reference words that belong to named entities.

    This avoids the earlier bug where a token string like "john" would mark
    every occurrence of "john" as an entity regardless of position.
    """
    doc = nlp(text)

    # Build normalised token sequence from spaCy tokens.
    norm_tokens = []
    spacy_to_norm_idx = {}

    for token in doc:
        norm = normalise_token(token.text)
        if not norm:
            continue
        pieces = norm.split()
        if not pieces:
            continue

        # Most tokens become one normalised token. Keep simple mapping.
        start_idx = len(norm_tokens)
        norm_tokens.extend(pieces)
        spacy_to_norm_idx[token.i] = list(range(start_idx, start_idx + len(pieces)))

    ner_indices = set()

    for ent in doc.ents:
        if ent.label_ not in NER_ENTITY_LABELS:
            continue

        for token in ent:
            token_lower = normalise_token(token.text)

            if not token_lower:
                continue
            if token_lower in NER_SPAN_EXCLUSIONS:
                continue
            if token.pos_ in {"DET", "ADP", "CCONJ", "SCONJ"}:
                continue

            for idx in spacy_to_norm_idx.get(token.i, []):
                ner_indices.add(idx)

    return ner_indices


# ── Categorisation ───────────────────────────────────────────────────────────

def categorise_word(word: str, word_idx: int, ner_indices: set, glossary: set) -> str:
    word_lower = word.lower()

    if word_idx in ner_indices:
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


# ── Alignment / error extraction ─────────────────────────────────────────────

NEGATION_DIALECT_PAIRS = {
    "no": {"not", "no", "n't"},
    "nae": {"not", "no", "n't"},
    "wisnae": {"wasn't", "wasnt"},
    "wasnae": {"wasn't", "wasnt"},
    "didnae": {"didn't", "didnt"},
    "dinnae": {"don't", "dont"},
    "cannae": {"can't", "cant"},
    "willnae": {"won't", "wont"},
    "wouldnae": {"wouldn't", "wouldnt"},
    "couldnae": {"couldn't", "couldnt"},
    "shouldnae": {"shouldn't", "shouldnt"},
    "disnae": {"doesn't", "doesnt"},
    "hasnae": {"hasn't", "hasnt"},
    "havenae": {"haven't", "havent"},
    "isnae": {"isn't", "isnt"},
    "arenae": {"aren't", "arent"},
}


def is_negation_form_change(ref_word: str, hyp_words: list) -> bool:
    """
    True only if the hypothesis preserves negation.

    Example:
        didnae -> didn't      preserved
        wisnae -> wasn't      preserved
        wisnae -> was         NOT preserved
    """
    ref_lower = ref_word.lower()
    hyp_lower_set = {h.lower() for h in hyp_words}

    if not hyp_lower_set:
        return False

    if hyp_lower_set & NEGATION_WORDS:
        return True

    equivalents = NEGATION_DIALECT_PAIRS.get(ref_lower, set())
    if hyp_lower_set & equivalents:
        return True

    return False


def get_error_ref_words(ref: str, hyp: str):
    """
    Return:
        ref_words: normalised reference words
        error_items: list of (ref_word_index, ref_word, hyp_words_at_position)

    Counts substitutions and deletions as errors.

    Important:
        If hyp is empty, all reference words are treated as deletions.
    """
    ref_norm = normalise(ref)
    hyp_norm = normalise(hyp)

    if not ref_norm:
        return [], []

    ref_words = ref_norm.split()

    if not hyp_norm:
        return ref_words, [(i, w, []) for i, w in enumerate(ref_words)]

    try:
        output = process_words(ref_norm, hyp_norm)
    except Exception:
        return ref_words, [(i, w, []) for i, w in enumerate(ref_words)]

    hyp_words = hyp_norm.split()
    error_items = []

    for chunk in output.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            r_words = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
            h_words = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

            for offset, w in enumerate(r_words):
                ref_idx = chunk.ref_start_idx + offset
                error_items.append((ref_idx, w, h_words))

    return ref_words, error_items


# ── Sampling helpers ─────────────────────────────────────────────────────────

def _load_samples(model, dataset, split="all"):
    filename = SUBSET_FILES.get((model, dataset))
    if not filename:
        print(f"No subset file found for model={model}, dataset={dataset}")
        return []

    path = os.path.join(SUBSETS_DIR, filename)
    if not os.path.isfile(path):
        print(f"No subset file found for model={model}, dataset={dataset}")
        return []

    with open(path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if split == "all":
        return samples

    allowed = set(get_indices_for_split(dataset, split))
    return [
        sample for sample in samples
        if sample.get("sample_index") in allowed
    ]


def sample_category_errors(model, dataset, category, nlp, glossary, n=5, split="all"):
    samples = _load_samples(model, dataset, split)
    shown = 0

    for s in samples:
        if shown >= n:
            break

        ref = s.get("ref", "")
        hyp = s.get("hyp", "")

        if not ref:
            continue

        ner_indices = get_ner_word_indices(nlp, ref)
        ref_words, error_items = get_error_ref_words(ref, hyp)

        if not ref_words:
            continue

        cat_errors = []

        for ref_idx, ref_word, hyp_words in error_items:
            cat = categorise_word(ref_word, ref_idx, ner_indices, glossary)

            if cat != category:
                continue

            if cat == "negation" and is_negation_form_change(ref_word, hyp_words):
                continue

            cat_errors.append(f"{ref_word} -> {hyp_words}")

        if cat_errors:
            print(f"\nSample {shown + 1}:")
            print(f"  REF: {ref[:200]}")
            print(f"  HYP: {hyp[:200]}")
            print(f"  {category} errors: {cat_errors}")
            shown += 1


def sample_category_total(model, dataset, category, nlp, glossary, n=5, split="all"):
    samples = _load_samples(model, dataset, split)
    shown = 0

    for s in samples:
        if shown >= n:
            break

        ref = s.get("ref", "")

        if not ref:
            continue

        ref_norm = normalise(ref)
        ref_words = ref_norm.split()
        ner_indices = get_ner_word_indices(nlp, ref)

        cat_words = [
            w for i, w in enumerate(ref_words)
            if categorise_word(w, i, ner_indices, glossary) == category
        ]

        if cat_words:
            print(f"\nSample {shown + 1}:")
            print(f"  REF: {ref[:200]}")
            print(f"  {category} words found: {cat_words}")
            shown += 1


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_model_dataset(model, dataset, nlp, glossary, split="all"):
    samples = _load_samples(model, dataset, split)
    if not samples:
        return None

    category_totals = defaultdict(int)
    category_errors = defaultdict(int)

    for s in samples:
        ref = s.get("ref", "")
        hyp = s.get("hyp", "")

        if not ref:
            continue

        ner_indices = get_ner_word_indices(nlp, ref)
        ref_words, error_items = get_error_ref_words(ref, hyp)

        if not ref_words:
            continue

        for i, w in enumerate(ref_words):
            cat = categorise_word(w, i, ner_indices, glossary)
            category_totals[cat] += 1

        for ref_idx, ref_word, hyp_words in error_items:
            cat = categorise_word(ref_word, ref_idx, ner_indices, glossary)

            if cat == "negation" and is_negation_form_change(ref_word, hyp_words):
                continue

            category_errors[cat] += 1

    profile = {}

    for cat in CATEGORIES:
        total = category_totals.get(cat, 0)
        errors = category_errors.get(cat, 0)
        rate = errors / total if total > 0 else None

        profile[cat] = {
            "errors": errors,
            "total": total,
            "rate": rate,
        }

    return profile


def print_profile(model, dataset, profile):
    print(f"\n── {model} / {dataset} ─────────────────────────────────")
    print(f"  {'Category':<22} {'Errors':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'-' * 52}")

    for cat in CATEGORIES:
        p = profile[cat]
        rate_str = f"{p['rate'] * 100:.1f}%" if p["rate"] is not None else "—"
        print(f"  {cat:<22} {p['errors']:>8} {p['total']:>8} {rate_str:>8}")


def print_rankings(all_profiles, dataset, models):
    print(f"\n{'=' * 60}")
    print(f"RANKINGS — {dataset} | lower error rate = more reliable")
    print(f"{'=' * 60}")

    for cat in CATEGORIES:
        rows = []

        for model in models:
            profile = all_profiles.get((model, dataset))
            if not profile:
                continue

            p = profile[cat]

            if p["total"] >= 5 and p["rate"] is not None:
                rows.append((model, p["rate"], p["total"]))

        if not rows:
            print(f"  {cat:<22} insufficient data")
            continue

        rows.sort(key=lambda x: x[1])
        ranking_str = " < ".join(
            f"{m}({r * 100:.1f}%, n={n})" for m, r, n in rows
        )
        print(f"  {cat:<22} {ranking_str}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="all")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
        help="Restrict profiles to one split. Use dev when deriving prompt rules.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        choices=CATEGORIES,
        help="Sample N items from this category. Use with --model and --n.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Must match a discovered subset file.",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="errors",
        choices=["errors", "total"],
        help="'errors' shows category error words; 'total' shows all words in category.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered models/datasets, then exit.",
    )

    args = parser.parse_args()

    if args.list:
        print(f"Discovered models:   {MODELS}")
        print(f"Discovered datasets: {DATASETS}")
        print("\nFiles found:")
        for (m, d), fname in sorted(SUBSET_FILES.items()):
            print(f"  ({m}, {d}) -> {fname}")
        return

    if not MODELS:
        print(
            f"No subset files discovered in {SUBSETS_DIR}. "
            "Expected filenames like '{model}_{dataset}_sub150.json'."
        )
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
            sample_category_errors(
                model, dataset, args.sample, nlp, glossary, args.n, args.split
            )
        else:
            sample_category_total(
                model, dataset, args.sample, nlp, glossary, args.n, args.split
            )

        return

    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    all_profiles = {}

    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"No subset files found for dataset='{dataset}'. Skipping.")
            continue

        for model in MODELS:
            profile = run_model_dataset(model, dataset, nlp, glossary, args.split)

            if profile:
                all_profiles[(model, dataset)] = profile
                print_profile(model, dataset, profile)

        print_rankings(all_profiles, dataset, MODELS)

    output = {
        f"{m}_{d}": p
        for (m, d), p in all_profiles.items()
    }

    os.makedirs("results/eval_suite", exist_ok=True)
    suffix = "" if args.split == "all" else f"_{args.split}"
    out_path = f"results/eval_suite/error_profiles{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
