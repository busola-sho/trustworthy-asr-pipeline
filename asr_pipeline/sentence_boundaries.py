"""Frozen content-preserving SaT boundary postprocessing."""

import re


SHORT_FRAGMENT_MAX_WORDS = 4

SENTENCE_FINAL_PUNCT = re.compile(r'[.!?]["\')]?\s*$')

MID_SEGMENT_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\u201c\u2018])')

OPENING_QUOTES = ('"', "'", "\u201c", "\u2018")

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "rev",
    "st",
    "jr",
    "sr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "mt",
    "ft",
    "no",
    "capt",
    "sgt",
    "col",
    "gen",
    "lt",
    "cpl",
    "maj",
}

INITIAL_OR_ACRONYM = re.compile(r"(?:\b[A-Z]\.)+\s*$")


def _is_abbreviation_before(text_before_period):
    if INITIAL_OR_ACRONYM.search(text_before_period):
        return True
    match = re.search(r"(\w+)\.\s*$", text_before_period)
    if not match:
        return False
    return match.group(1).lower() in ABBREVIATIONS


def rule_a_split_merged_sentences(segments):
    """Returns (result_segments, split_events) - split_events is a list
    of (original_merged_text, [resulting_pieces]) for EVERY actual
    split made, so each can be individually labeled/logged."""
    result = []
    split_events = []
    for segment in segments:
        stripped = segment.strip()
        pieces = []
        last_end = 0
        for match in MID_SEGMENT_BOUNDARY.finditer(stripped):
            candidate_before = stripped[: match.start()]
            if _is_abbreviation_before(candidate_before):
                continue
            pieces.append(stripped[last_end : match.start()].strip())
            last_end = match.start()
        pieces.append(stripped[last_end:].strip())
        pieces = [p for p in pieces if p]

        if len(pieces) > 1:
            split_events.append((stripped, pieces))

        result.extend(p + " " for p in pieces)
    return result, split_events


def rule_b_merge_reporting_quotation(segments):
    """Returns (result_segments, merge_events) - merge_events is a list
    of (before_seg, after_seg, merged_result) for EVERY actual merge."""
    if not segments:
        return segments, []
    result = [segments[0]]
    merge_events = []
    for segment in segments[1:]:
        prev = result[-1]
        prev_ends_open = prev.rstrip().endswith((",", ":"))
        curr_starts_quote = segment.lstrip().startswith(OPENING_QUOTES)
        if prev_ends_open and curr_starts_quote:
            merged = prev.rstrip() + " " + segment.strip() + " "
            merge_events.append((prev, segment, merged))
            result[-1] = merged
        else:
            result.append(segment)
    return result, merge_events


def flag_short_fragments(segments):
    flagged = []
    for i, segment in enumerate(segments):
        word_count = len(segment.split())
        has_final_punct = bool(SENTENCE_FINAL_PUNCT.search(segment.strip()))
        if word_count < SHORT_FRAGMENT_MAX_WORDS and not has_final_punct:
            flagged.append((i, segment))
    return flagged


def normalise_for_boundary_check(segments):
    return re.sub(r"\s+", " ", "".join(segments)).strip()


def apply_postprocessing(raw_segments):
    """Returns (final_segments, split_events, merge_events)."""
    after_a, split_events = rule_a_split_merged_sentences(raw_segments)
    after_b, merge_events = rule_b_merge_reporting_quotation(after_a)

    raw_text = normalise_for_boundary_check(raw_segments)
    processed_text = normalise_for_boundary_check(after_b)
    if raw_text != processed_text:
        raise ValueError(
            "Post-processing altered transcript content - this must never happen. "
            f"RAW: {raw_text!r}\nPROCESSED: {processed_text!r}"
        )
    return after_b, split_events, merge_events


def segment_hypothesis(hypothesis, sentence_splitter):
    """Apply the frozen SaT and boundary rules to hypothesis text alone."""
    raw_segments = sentence_splitter.split(hypothesis)
    segments, _, _ = apply_postprocessing(raw_segments)
    return [segment.strip() for segment in segments if segment.strip()]
