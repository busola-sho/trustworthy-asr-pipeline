"""
label_disagreements.py

For each clip, aligns Qwen (anchor) against Whisper and Parakeet word by word,
finds disagreement positions, and labels each with linguistic metadata:
  - dialect: word found in Scottish glossary
  - named_entity: word is part of a named entity (spaCy NER)
  - number: word is a digit or written-out number
  - negation: word is a negation marker
  - profanity: word is mild profanity / informal expression
  - function: common function word
  - content: general content word

Output per clip:
  {
    "clip_index": 5,
    "ref": "...",
    "qwen_hyp": "...",
    "disagreements": [
      {
        "position": 7,
        "qwen_word": "was",
        "whisper_word": "wisnae",
        "parakeet_word": "wisnae",
        "label": "dialect",
        "scottish_meaning": "was not (negation)",
        "n_agree_on_alt": 2,
        "consensus_alt": "wisnae"
      }
    ]
  }

Usage:
    python scripts/label_disagreements.py --dataset commonvoice --n 20
    python scripts/label_disagreements.py --dataset commonvoice --output analysis/disagreements_commonvoice.json
"""

import json
import os
import re
import argparse
from collections import Counter
from jiwer import process_words

BENCHMARKS_DIR  = "benchmarks"
GLOSSARY_PATH   = "data/scottish_glossary.json"

CANONICAL_FILES = {
    ("qwen",     "commonvoice"):      "qwen_commonvoice_20260524_153426.json",
    ("qwen",     "edacc"):            "qwen_edacc_20260525_204314.json",
    ("qwen",     "english_dialects"): "qwen_english_dialects_20260525_000627.json",
    ("qwen",     "shetland"):         "shetland_qwen3asr_20260603_150124.json",
    ("whisper",  "commonvoice"):      "whisper_commonvoice_20260524_083930.json",
    ("whisper",  "edacc"):            "whisper_edacc_20260525_184504.json",
    ("whisper",  "english_dialects"): "whisper_english_dialects_20260525_110315.json",
    ("whisper",  "shetland"):         "shetland_whisper_20260603_123115.json",
    ("parakeet", "commonvoice"):      "parakeet_commonvoice_20260524_150129.json",
    ("parakeet", "edacc"):            "parakeet_edacc_20260525_184006.json",
    ("parakeet", "english_dialects"): "parakeet_english_dialects_20260524_234807.json",
    ("parakeet", "shetland"):         "shetland_parakeet_20260606_134131.json",
}

ALTERNATIVE_MODELS = ["whisper", "parakeet"]

# ── Linguistic resources ───────────────────────────────────────────────────────

NEGATIONS = {
    "not", "no", "never", "nobody", "nothing", "nowhere", "neither",
    "nor", "nae", "nope", "dinnae", "didnae", "wasnae", "cannae",
    "wouldnae", "couldnae", "shouldnae", "havenae", "isnae", "doesnae",
    "amna", "arnae", "willnae", "mightna", "needna", "darenae",
    "wisnae", "hadnae", "hasnae",
}

MILD_PROFANITY = {
    "bloody", "damn", "hell", "crap", "arse", "shite", "sod", "blimey",
    "bugger", "bleeding", "blooming", "ruddy", "blinking", "flipping",
    "effing", "frigging", "fecking", "feck", "keech", "shit", "bastard",
    "bawbag", "numpty",
}

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "not", "so", "as", "if", "then", "than", "when", "where", "which",
    "who", "what", "how", "there", "here", "just", "also", "up", "out",
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "first", "second", "third", "fourth",
    "fifth", "half", "quarter",
}

# Common English words that appear in the glossary but should NOT be labelled dialect
# (they have secondary Scottish meanings but are primarily standard English)
GLOSSARY_SKIP = {
    "like", "she", "he", "it", "air", "an", "as", "at", "be", "by",
    "do", "go", "in", "is", "me", "my", "no", "of", "on", "or", "so",
    "to", "up", "us", "we", "a", "i", "ill", "ail", "ark", "art",
    "set", "one", "two", "ten", "even", "ever", "own", "age", "aim",
    "arm", "ask", "ate", "aye", "ken", "an",
}

def load_glossary() -> dict:
    if not os.path.exists(GLOSSARY_PATH):
        print(f"  WARNING: glossary not found at {GLOSSARY_PATH}")
        print(f"  Run: python scripts/scrape_scottish_glossary.py")
        return {}
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)

def clean_word(w: str) -> str:
    """Lowercase and strip punctuation for lookup."""
    return re.sub(r"[^\w']", "", w.lower()).strip("'")

def is_number(w: str) -> bool:
    w = clean_word(w)
    return w.isdigit() or w in NUMBER_WORDS

def label_word(word: str, alt_words: list, glossary: dict,
               spacy_ents: list = None) -> tuple:
    """
    Returns (label, detail) for a disagreement position.
    Checks the word AND its alternatives against all categories.
    """
    all_words = [word] + [w for w in alt_words if w and w != "[DEL]"]
    cleaned   = [clean_word(w) for w in all_words]

    # check named entity from spaCy
    if spacy_ents:
        word_lower = clean_word(word)
        for ent_text, ent_label in spacy_ents:
            if word_lower in clean_word(ent_text).split():
                return "named_entity", f"{ent_label}: {ent_text}"

    # check dialect glossary — any of the words
    # skip words that are common English but happen to be in the glossary
    for w, c in zip(all_words, cleaned):
        if c in glossary and c not in GLOSSARY_SKIP and len(c) > 2:
            return "dialect", glossary[c]

    # check negation
    for c in cleaned:
        if c in NEGATIONS:
            return "negation", f"negation marker: {c}"

    # check number
    for w in all_words:
        if is_number(w):
            return "number", f"numeric: {w}"

    # check mild profanity
    for c in cleaned:
        if c in MILD_PROFANITY:
            return "profanity", f"mild profanity: {c}"

    # check function word
    if all(c in FUNCTION_WORDS for c in cleaned if c):
        return "function", "common function word"

    return "content", "general content word"

# ── Alignment ──────────────────────────────────────────────────────────────────

def align_to_qwen(qwen_hyp: str, alt_hyp: str) -> dict:
    """
    Aligns alt_hyp to qwen_hyp. Returns dict: qwen_position -> alt_word.
    None = deletion, actual word = substitution or equal.
    """
    try:
        out       = process_words(qwen_hyp, alt_hyp)
        qwen_words = out.references[0]
        alt_words  = out.hypotheses[0]
        alignments = out.alignments[0]

        result = {i: None for i in range(len(qwen_words))}

        for op in alignments:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = alt_words[hyp_idx] if hyp_idx < len(alt_words) else None
            elif op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                    result[i] = alt_words[hyp_idx] if hyp_idx < len(alt_words) else None
            elif op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    result[i] = "[DEL]"

        return qwen_words, result
    except Exception:
        return [], {}

def find_labelled_disagreements(qwen_hyp: str, alt_hyps: dict,
                                 glossary: dict, spacy_nlp=None) -> list:
    """
    Finds all disagreement positions and labels them linguistically.
    """
    # get spaCy named entities from Qwen transcript
    spacy_ents = []
    if spacy_nlp:
        doc = spacy_nlp(qwen_hyp)
        spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]

    # align each alternative to Qwen
    alignments = {}
    qwen_words = None
    for model, hyp in alt_hyps.items():
        words, alignment = align_to_qwen(qwen_hyp, hyp)
        alignments[model] = alignment
        if qwen_words is None:
            qwen_words = words

    if not qwen_words:
        return []

    disagreements = []

    for i, qwen_word in enumerate(qwen_words):
        qwen_clean = clean_word(qwen_word)

        # skip empty tokens (punctuation-only)
        if not qwen_clean:
            continue

        alt_words_at_pos = {}
        n_disagree = 0

        for model, alignment in alignments.items():
            alt_word  = alignment.get(i, "[DEL]") or "[DEL]"
            alt_clean = clean_word(alt_word)
            alt_words_at_pos[model] = alt_word

            # compare cleaned versions — ignores punctuation differences
            if alt_clean != qwen_clean and not (alt_clean == "" and qwen_clean == ""):
                n_disagree += 1

        if n_disagree == 0:
            continue  # all models agree with Qwen — no disagreement

        # find consensus alternative — only count genuine substitutions
        non_qwen = [
            clean_word(w) for w in alt_words_at_pos.values()
            if clean_word(w) != qwen_clean
            and clean_word(w) != ""
            and w != "[DEL]"
        ]
        alt_counts    = Counter(non_qwen)
        consensus_alt = alt_counts.most_common(1)[0][0] if alt_counts else None
        n_agree_alt   = alt_counts.most_common(1)[0][1] if alt_counts else 0

        # classify disagreement type
        all_del      = all(v == "[DEL]" for v in alt_words_at_pos.values() if clean_word(v) != qwen_clean)
        any_sub      = any(v != "[DEL]" and clean_word(v) != qwen_clean for v in alt_words_at_pos.values())
        disagree_type = "deletion" if all_del else "substitution" if any_sub else "mixed"

        # strong consensus = both alternatives agree on same word (not deletion)
        strong_consensus = n_agree_alt >= 2 and consensus_alt and consensus_alt != "[del]"

        # label
        all_alts = list(alt_words_at_pos.values())
        label, detail = label_word(qwen_word, all_alts, glossary, spacy_ents)

        disagreements.append({
            "position":         i,
            "qwen_word":        qwen_word,
            "alternatives":     alt_words_at_pos,
            "n_disagree":       n_disagree,
            "consensus_alt":    consensus_alt,
            "n_agree_alt":      n_agree_alt,
            "strong_consensus": strong_consensus,
            "disagree_type":    disagree_type,
            "label":            label,
            "detail":           detail,
        })

    return disagreements

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="commonvoice",
                        choices=["commonvoice", "edacc", "english_dialects", "shetland"])
    parser.add_argument("--n",       type=int, default=None,
                        help="Number of clips to process (default: all)")
    parser.add_argument("--output",  default=None,
                        help="Output JSON path")
    parser.add_argument("--no-spacy", action="store_true",
                        help="Skip spaCy NER (faster but no named entity detection)")
    args = parser.parse_args()

    # load glossary
    glossary = load_glossary()
    print(f"Loaded {len(glossary)} Scottish words from glossary")

    # load spaCy
    spacy_nlp = None
    if not args.no_spacy:
        try:
            import spacy
            spacy_nlp = spacy.load("en_core_web_sm")
            print("spaCy NER loaded")
        except Exception as e:
            print(f"  WARNING: spaCy not available ({e}) — skipping NER")
            print(f"  Install with: pip install spacy && python -m spacy download en_core_web_sm")

    # load model samples
    model_samples = {}
    for model in ["qwen"] + ALTERNATIVE_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, args.dataset)])
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    n = len(model_samples["qwen"])
    if args.n:
        n = min(n, args.n)

    print(f"\nProcessing {n} clips from {args.dataset}...")

    results = []
    label_counts = Counter()

    for i in range(n):
        ref      = model_samples["qwen"][i]["ref"]
        qwen_hyp = model_samples["qwen"][i]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        alt_hyps = {m: model_samples[m][i]["hyp"] for m in ALTERNATIVE_MODELS}

        disagreements = find_labelled_disagreements(
            qwen_hyp, alt_hyps, glossary, spacy_nlp
        )

        for d in disagreements:
            label_counts[d["label"]] += 1

        results.append({
            "clip_index":    i,
            "ref":           ref,
            "qwen_hyp":      qwen_hyp,
            "n_disagreements": len(disagreements),
            "disagreements": disagreements,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} done")

    # summary
    total_disagreements  = sum(r["n_disagreements"] for r in results)
    total_strong         = sum(
        1 for r in results for d in r["disagreements"] if d.get("strong_consensus")
    )
    total_substitutions  = sum(
        1 for r in results for d in r["disagreements"] if d.get("disagree_type") == "substitution"
    )
    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Clips processed:               {len(results)}")
    print(f"  Total disagreements:           {total_disagreements}")
    print(f"  Substitutions:                 {total_substitutions} ({total_substitutions/total_disagreements*100:.1f}%)")
    print(f"  Strong consensus (both agree): {total_strong} ({total_strong/total_disagreements*100:.1f}%) ← actionable")
    print(f"\n  Label distribution (all disagreements):")
    for label, count in label_counts.most_common():
        pct = count / total_disagreements * 100
        print(f"    {label:<15} {count:>6} ({pct:>5.1f}%)")

    # save
    os.makedirs("analysis", exist_ok=True)
    out_path = args.output or f"analysis/disagreements_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    # show a few examples
    print(f"\n── Sample disagreements ──────────────────────────────")
    shown = 0
    for r in results:
        if shown >= 3:
            break
        if not r["disagreements"]:
            continue
        print(f"\n  Clip {r['clip_index']}:")
        print(f"  QWEN: {r['qwen_hyp'][:100]}")
        for d in r["disagreements"][:3]:
            alts = " | ".join(f"{m}={v}" for m, v in d["alternatives"].items())
            print(f"    pos={d['position']} qwen='{d['qwen_word']}' [{d['label']}] → {alts}")
            print(f"    detail: {d['detail']}")
        shown += 1

if __name__ == "__main__":
    main()