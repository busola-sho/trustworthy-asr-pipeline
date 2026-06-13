"""
label_disagreements_phrase.py

Hybrid phrase+word level alignment:
1. Phrase-level: finds disagreement REGIONS (robust to alignment slips)
2. Word-level within region: finds the specific changed words
3. Filters pure case/punctuation differences
4. Labels changed words linguistically

Usage:
    python scripts/label_disagreements_phrase.py --dataset commonvoice --n 20 --no-spacy
    python scripts/label_disagreements_phrase.py --dataset commonvoice
"""

import json
import os
import re
import argparse
from collections import Counter
from jiwer import process_words

BENCHMARKS_DIR = "benchmarks"
GLOSSARY_PATH  = "data/scottish_glossary.json"

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
    "amna", "arnae", "willnae", "wisnae", "hadnae", "hasnae",
}

MILD_PROFANITY = {
    "bloody", "damn", "hell", "crap", "arse", "shite", "sod", "blimey",
    "bugger", "bleeding", "blooming", "ruddy", "blinking", "flipping",
    "effing", "frigging", "fecking", "feck", "keech", "bastard",
    "bawbag", "numpty",
}

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "that", "this", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "so", "as", "if", "then", "than", "when", "where", "which",
    "who", "what", "how", "there", "here", "just", "also",
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "first", "second", "third",
    "half", "quarter",
}

GLOSSARY_SKIP = {
    "like", "she", "he", "it", "air", "an", "as", "at", "be", "by",
    "do", "go", "in", "is", "me", "my", "no", "of", "on", "or", "so",
    "to", "up", "us", "we", "a", "i", "ill", "ail", "ark", "art",
    "set", "one", "two", "ten", "even", "ever", "own", "age", "aim",
    "arm", "ask", "ate", "aye", "ken",
}

# ── Utils ──────────────────────────────────────────────────────────────────────

def load_glossary() -> dict:
    if not os.path.exists(GLOSSARY_PATH):
        print(f"  WARNING: glossary not found — run scrape_scottish_glossary.py")
        return {}
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)

def clean_word(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower()).strip("'")

def label_word(w: str, glossary: dict, spacy_ents: list = None) -> tuple:
    """Returns (label, detail)."""
    c = clean_word(w)
    if not c:
        return "punctuation", ""
    if spacy_ents:
        for ent_text, ent_label in spacy_ents:
            if c in clean_word(ent_text).split():
                return "named_entity", f"{ent_label}: {ent_text}"
    if c in glossary and c not in GLOSSARY_SKIP and len(c) > 2:
        return "dialect", glossary[c]
    if c in NEGATIONS:
        return "negation", f"negation marker: {c}"
    if w.isdigit() or c in NUMBER_WORDS:
        return "number", f"numeric: {w}"
    if c in MILD_PROFANITY:
        return "profanity", f"mild profanity: {c}"
    if c in FUNCTION_WORDS:
        return "function", "common function word"
    return "content", "general content word"

# ── Alignment ──────────────────────────────────────────────────────────────────

def align(ref_hyp: str, alt_hyp: str):
    """Returns (ref_words, alt_words, ops)."""
    try:
        out = process_words(ref_hyp, alt_hyp)
        return out.references[0], out.hypotheses[0], out.alignments[0]
    except Exception:
        return [], [], []

def get_disagreement_mask(qwen_words, alt_words, ops):
    """
    Boolean mask over qwen positions.
    True = genuine difference (after cleaning).
    False = same content, just case/punctuation.
    """
    mask = [False] * len(qwen_words)
    for op in ops:
        if op.type == "equal":
            continue
        elif op.type == "substitute":
            for i in range(op.ref_start_idx, op.ref_end_idx):
                hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                qw = clean_word(qwen_words[i])
                aw = clean_word(alt_words[hyp_idx]) if hyp_idx < len(alt_words) else ""
                if qw != aw:  # genuine difference after cleaning
                    mask[i] = True
        elif op.type == "delete":
            for i in range(op.ref_start_idx, op.ref_end_idx):
                mask[i] = True
        elif op.type == "insert":
            pos = min(op.ref_start_idx, len(qwen_words) - 1)
            mask[pos] = True
    return mask

def get_alt_words_at_positions(qwen_words, alt_words, ops, start, end):
    """
    Returns a list of (qwen_pos, alt_word_or_None) for positions in [start, end).
    Also returns any inserted words.
    """
    # position -> alt word
    pos_map = {}
    insertions = []  # (after_pos, word)

    for op in ops:
        if op.type == "equal" or op.type == "substitute":
            for i in range(max(op.ref_start_idx, start), min(op.ref_end_idx, end)):
                hyp_idx = op.hyp_start_idx + (i - op.ref_start_idx)
                pos_map[i] = alt_words[hyp_idx] if hyp_idx < len(alt_words) else None
        elif op.type == "delete":
            for i in range(max(op.ref_start_idx, start), min(op.ref_end_idx, end)):
                pos_map[i] = "[DEL]"
        elif op.type == "insert":
            if start <= op.ref_start_idx <= end:
                for hyp_idx in range(op.hyp_start_idx, op.hyp_end_idx):
                    if hyp_idx < len(alt_words):
                        insertions.append((op.ref_start_idx, alt_words[hyp_idx]))

    return pos_map, insertions

def group_into_regions(mask, context=1):
    """Groups True positions into (start, end) regions with padding."""
    regions = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = max(0, i - context)
            while i < n and mask[i]:
                i += 1
            end = min(n, i + context)
            if regions and start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], end)
            else:
                regions.append((start, end))
        else:
            i += 1
    return regions

# ── Core analysis ──────────────────────────────────────────────────────────────

def analyse_clip(qwen_hyp, alt_hyps, glossary, spacy_nlp=None):
    spacy_ents = []
    if spacy_nlp:
        doc = spacy_nlp(qwen_hyp)
        spacy_ents = [(ent.text, ent.label_) for ent in doc.ents]

    # align each model to qwen
    model_data = {}
    qwen_words = None
    for model, hyp in alt_hyps.items():
        q_words, a_words, ops = align(qwen_hyp, hyp)
        model_data[model] = (q_words, a_words, ops)
        if qwen_words is None:
            qwen_words = q_words

    if not qwen_words:
        return []

    # build combined disagreement mask
    combined_mask = [False] * len(qwen_words)
    for model, (q_words, a_words, ops) in model_data.items():
        mask = get_disagreement_mask(q_words, a_words, ops)
        for i, v in enumerate(mask):
            if i < len(combined_mask) and v:
                combined_mask[i] = True

    # group into regions
    raw_regions = group_into_regions(combined_mask, context=1)

    results = []
    for start, end in raw_regions:
        qwen_phrase = " ".join(qwen_words[start:end])

        # find changed words within region per model
        model_changes = {}
        for model, (q_words, a_words, ops) in model_data.items():
            pos_map, insertions = get_alt_words_at_positions(
                q_words, a_words, ops, start, end
            )
            # find genuine word-level changes within region
            changes = []
            for i in range(start, end):
                qw = qwen_words[i] if i < len(qwen_words) else ""
                aw = pos_map.get(i)
                if aw is None:
                    continue
                qw_c = clean_word(qw)
                aw_c = clean_word(aw) if aw != "[DEL]" else "[DEL]"
                if qw_c != aw_c:
                    changes.append({
                        "position":  i,
                        "qwen_word": qw,
                        "alt_word":  aw,
                    })
            # add insertions
            for pos, ins_word in insertions:
                changes.append({
                    "position":  pos,
                    "qwen_word": "[INS]",
                    "alt_word":  ins_word,
                    "insertion": True,
                })
            model_changes[model] = changes

        # skip region if no genuine changes found (pure case/punct)
        if not any(model_changes.values()):
            continue

        # label each changed word
        labelled_changes = {}
        for model, changes in model_changes.items():
            labelled = []
            for ch in changes:
                # label both qwen word and alt word, pick most informative
                if ch.get("insertion"):
                    label, detail = label_word(ch["alt_word"], glossary, spacy_ents)
                else:
                    ql, qd = label_word(ch["qwen_word"], glossary, spacy_ents)
                    al, ad = label_word(ch["alt_word"],  glossary, spacy_ents)
                    # prefer more specific label
                    priority = ["named_entity", "dialect", "negation", "number",
                                "profanity", "content", "function", "punctuation"]
                    label  = ql if priority.index(ql) <= priority.index(al) else al
                    detail = qd if priority.index(ql) <= priority.index(al) else ad
                labelled.append({**ch, "label": label, "detail": detail})
            labelled_changes[model] = labelled

        # aggregate flags
        all_labels = [
            ch["label"]
            for changes in labelled_changes.values()
            for ch in changes
        ]
        dialect_words  = list({ch["alt_word"] for changes in labelled_changes.values()
                               for ch in changes if ch["label"] == "dialect"})
        named_entities = list({ch["alt_word"] for changes in labelled_changes.values()
                               for ch in changes if ch["label"] == "named_entity"})
        has_negation   = "negation"   in all_labels
        has_number     = "number"     in all_labels
        has_profanity  = "profanity"  in all_labels

        # check strong consensus — both models agree on same alt for same position
        # find positions where both models have a change
        positions_changed = {}
        for model, changes in labelled_changes.items():
            for ch in changes:
                if not ch.get("insertion"):
                    pos = ch["position"]
                    if pos not in positions_changed:
                        positions_changed[pos] = {}
                    positions_changed[pos][model] = clean_word(ch["alt_word"])

        strong_consensus_positions = []
        for pos, model_alts in positions_changed.items():
            if len(model_alts) >= 2:
                alts = list(model_alts.values())
                if len(set(alts)) == 1 and alts[0] != clean_word(qwen_words[pos]):
                    strong_consensus_positions.append({
                        "position":  pos,
                        "qwen_word": qwen_words[pos],
                        "consensus": alts[0],
                    })

        dominant_label = Counter(all_labels).most_common(1)[0][0] if all_labels else "content"

        results.append({
            "region":                     [start, end],
            "qwen_phrase":                qwen_phrase,
            "model_changes":              labelled_changes,
            "strong_consensus_positions": strong_consensus_positions,
            "has_strong_consensus":       len(strong_consensus_positions) > 0,
            "dominant_label":             dominant_label,
            "dialect_words":              dialect_words,
            "named_entities":             named_entities,
            "has_negation":               has_negation,
            "has_number":                 has_number,
            "has_profanity":              has_profanity,
        })

    return results

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="commonvoice",
                        choices=["commonvoice", "edacc", "english_dialects", "shetland"])
    parser.add_argument("--n",        type=int, default=None)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--no-spacy", action="store_true")
    args = parser.parse_args()

    glossary = load_glossary()
    print(f"Loaded {len(glossary)} Scottish words")

    spacy_nlp = None
    if not args.no_spacy:
        try:
            import spacy
            spacy_nlp = spacy.load("en_core_web_sm")
            print("spaCy NER loaded")
        except Exception as e:
            print(f"  WARNING: spaCy not available — {e}")

    model_samples = {}
    for model in ["qwen"] + ALTERNATIVE_MODELS:
        path = os.path.join(BENCHMARKS_DIR, CANONICAL_FILES[(model, args.dataset)])
        with open(path) as f:
            model_samples[model] = json.load(f)["samples"]

    n = min(args.n or len(model_samples["qwen"]), len(model_samples["qwen"]))
    print(f"Processing {n} clips from {args.dataset}...\n")

    results       = []
    total_regions = 0
    strong_total  = 0
    label_counts  = Counter()
    flag_counts   = Counter()

    for i in range(n):
        ref      = model_samples["qwen"][i]["ref"]
        qwen_hyp = model_samples["qwen"][i]["hyp"]

        if "IGNORE_TIME_SEGMENT_IN_SCORING" in ref:
            continue

        alt_hyps = {m: model_samples[m][i]["hyp"] for m in ALTERNATIVE_MODELS}
        regions  = analyse_clip(qwen_hyp, alt_hyps, glossary, spacy_nlp)

        for r in regions:
            total_regions += 1
            label_counts[r["dominant_label"]] += 1
            if r["has_strong_consensus"]:
                strong_total += 1
            if r["dialect_words"]:
                flag_counts["has_dialect"] += 1
            if r["named_entities"]:
                flag_counts["has_named_entity"] += 1
            if r["has_negation"]:
                flag_counts["has_negation"] += 1
            if r["has_number"]:
                flag_counts["has_number"] += 1

        results.append({
            "clip_index":          i,
            "ref":                 ref,
            "qwen_hyp":            qwen_hyp,
            "n_regions":           len(regions),
            "disagreement_regions": regions,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} done")

    print(f"\n── Summary ──────────────────────────────────────────")
    print(f"  Clips:                {len(results)}")
    print(f"  Total regions:        {total_regions}")
    print(f"  Avg per clip:         {total_regions/len(results):.1f}")
    print(f"  Strong consensus:     {strong_total} ({strong_total/total_regions*100:.1f}%) ← actionable")
    print(f"\n  Region flags:")
    for key, count in flag_counts.most_common():
        print(f"    {key:<22} {count:>6} ({count/total_regions*100:>5.1f}%)")
    print(f"\n  Dominant label distribution:")
    for label, count in label_counts.most_common():
        print(f"    {label:<15} {count:>6} ({count/total_regions*100:>5.1f}%)")

    # show strong consensus examples
    print(f"\n── Strong consensus regions (sample) ────────────────")
    shown = 0
    for r in results:
        if shown >= 5:
            break
        for region in r["disagreement_regions"]:
            if region["has_strong_consensus"]:
                print(f"\n  Clip {r['clip_index']} | phrase: '{region['qwen_phrase']}'")
                for sc in region["strong_consensus_positions"]:
                    print(f"    pos={sc['position']} qwen='{sc['qwen_word']}' → both say '{sc['consensus']}'")
                for model, changes in region["model_changes"].items():
                    if changes:
                        ch_str = ", ".join(
                            f"'{c['qwen_word']}'→'{c['alt_word']}' [{c['label']}]"
                            for c in changes
                        )
                        print(f"    {model}: {ch_str}")
                if region["dialect_words"]:
                    print(f"    ⚑ Dialect: {region['dialect_words']}")
                if region["has_negation"]:
                    print(f"    ⚠ Negation")
                if region["has_number"]:
                    print(f"    # Number")
                shown += 1
                if shown >= 5:
                    break

    os.makedirs("analysis", exist_ok=True)
    out_path = args.output or f"analysis/phrase_disagreements_{args.dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()