"""
pipeline/ner_flag.py

Named entity flagging for the reviewer interface.

Runs spaCy NER on the final transcript to identify names, places,
organisations, dates, and other entities that are high-risk for
transcription errors in a policing context.

Flags are passed to the reviewer interface so the reviewer can
quickly check whether these high-stakes words were transcribed correctly.

Usage:
    from pipeline.ner_flag import flag_named_entities

    flags = flag_named_entities(
        "She met John Smith at Princes Street on the 14th of March."
    )
    # [
    #   {"text": "John Smith", "label": "PERSON", "start_char": 8, "end_char": 18},
    #   {"text": "Princes Street", "label": "LOC", "start_char": 22, "end_char": 36},
    #   {"text": "14th of March", "label": "DATE", "start_char": 44, "end_char": 57},
    # ]
"""

from typing import List, Dict, Optional

# high-risk entity types for policing context
HIGH_RISK_LABELS = {
    "PERSON",   # names of people (witness, suspect, officer)
    "ORG",      # organisations (police, court, companies)
    "GPE",      # geopolitical entities (cities, countries)
    "LOC",      # locations (streets, buildings)
    "FAC",      # facilities (prisons, hospitals)
    "DATE",     # dates (critical for timelines)
    "TIME",     # times (critical for alibis)
    "MONEY",    # amounts
    "LAW",      # laws, acts, regulations
    "EVENT",    # named events
}

_nlp = None  # lazy-loaded


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm...")
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def flag_named_entities(
    transcript: str,
    high_risk_only: bool = True,
) -> List[Dict]:
    """
    Run NER on transcript and return flagged entities.

    Args:
        transcript:     final combined transcript text
        high_risk_only: if True, only return HIGH_RISK_LABELS entities

    Returns:
        list of entity dicts sorted by position:
        [{"text", "label", "label_description", "start_char", "end_char"}, ...]
    """
    if not transcript or not transcript.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(transcript)

    LABEL_DESCRIPTIONS = {
        "PERSON":  "Person name",
        "ORG":     "Organisation",
        "GPE":     "City/Country",
        "LOC":     "Location",
        "FAC":     "Facility",
        "DATE":    "Date",
        "TIME":    "Time",
        "MONEY":   "Amount",
        "LAW":     "Law/Act",
        "EVENT":   "Named event",
        "NORP":    "Nationality/Group",
        "PRODUCT": "Product",
        "WORK_OF_ART": "Title",
    }

    flags = []
    for ent in doc.ents:
        if high_risk_only and ent.label_ not in HIGH_RISK_LABELS:
            continue
        flags.append({
            "text":              ent.text,
            "label":             ent.label_,
            "label_description": LABEL_DESCRIPTIONS.get(ent.label_, ent.label_),
            "start_char":        ent.start_char,
            "end_char":          ent.end_char,
        })

    return flags


def annotate_transcript(
    transcript: str,
    flags: Optional[List[Dict]] = None,
) -> str:
    """
    Return transcript with NER flags inserted inline for display.
    e.g. "She met [John Smith|PERSON] at [Princes Street|LOC]"

    Args:
        transcript: final transcript
        flags:      output of flag_named_entities() — computed if None

    Returns:
        annotated string
    """
    if flags is None:
        flags = flag_named_entities(transcript)

    if not flags:
        return transcript

    # insert annotations from end to start so char positions stay valid
    result = transcript
    for flag in sorted(flags, key=lambda f: f["start_char"], reverse=True):
        s, e   = flag["start_char"], flag["end_char"]
        label  = flag["label"]
        result = result[:s] + f"[{result[s:e]}|{label}]" + result[e:]

    return result