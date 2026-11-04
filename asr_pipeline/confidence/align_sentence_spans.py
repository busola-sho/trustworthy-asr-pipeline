"""Four-source sentence confidence: align_segments_hybrid."""

import re
import difflib

from asr_pipeline.parallel_calls import call_models_in_parallel

MECHANICAL_THRESHOLD = 0.55
WIDE_FUZZY_THRESHOLD = 0.90
MAX_EXTRA = 15
LOCAL_WINDOW_WORDS = 150
WIDE_WINDOW_WORDS = 500
MAX_SPAN_LENGTH_RATIO = 3.0
DEFAULT_MAX_WORKERS = 8

LLM_FALLBACK_PROMPT = """You are given a short excerpt from one transcript and ONE
sentence-like segment from another transcript of the same spoken audio.
Identify the MINIMAL span of text in the excerpt that corresponds to the
segment - normally just ONE sentence-like span, matching the segment's
own length and content. Do NOT include surrounding content.

Transcript excerpt:
{excerpt}

Target segment:
{segment}

Respond ONLY with that minimal corresponding span copied verbatim from
the excerpt. If there is no clear corresponding content, respond exactly:
NONE"""


def tokenize_with_offsets(text):
    tokens, spans = [], []
    for m in re.finditer(r"[\w']+", text):
        tokens.append(m.group(0).lower())
        spans.append((m.start(), m.end()))
    return tokens, spans


def segment_tokens(text):
    return re.findall(r"[\w']+", text.lower())


def segment_word_count(text):
    return len(segment_tokens(text))


def span_text(reference, reference_word_positions, start, end):
    return reference[
        reference_word_positions[start][0] : reference_word_positions[end - 1][1]
    ]


def mechanical_match(segment_text, reference_tokens, cursor):
    anchor = segment_tokens(segment_text)
    if not anchor:
        return None, 0.0, cursor

    n = len(anchor)
    min_len = max(1, n - MAX_EXTRA)
    max_len = n + MAX_EXTRA

    start0 = max(0, cursor)
    search_end = min(len(reference_tokens), start0 + max(n * 4 + MAX_EXTRA, 60))

    best_start = best_end = None
    best_score = 0.0

    for start in range(start0, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > len(reference_tokens):
                continue

            score = difflib.SequenceMatcher(
                None, anchor, reference_tokens[start:end]
            ).ratio()

            if score > best_score:
                best_score, best_start, best_end = score, start, end

        if best_start is not None and best_score >= 0.88 and start > best_start + 5:
            break

    return best_start, best_score, best_end


def global_exact_match(segment_text, reference_tokens):
    """
    Cheap whole-reference exact normalized-token rescue.
    Only accepts a UNIQUE exact occurrence.
    """
    anchor = segment_tokens(segment_text)
    if not anchor:
        return None

    n = len(anchor)
    matches = []

    for start in range(0, len(reference_tokens) - n + 1):
        if reference_tokens[start : start + n] == anchor:
            matches.append((start, start + n))
            if len(matches) > 1:
                return None

    return matches[0] if len(matches) == 1 else None


def wide_local_fuzzy_match(segment_text, reference_tokens, estimated_cursor):
    """
    Fuzzy rescue in a wide LOCAL window, not over the whole reference.
    """
    anchor = segment_tokens(segment_text)
    if not anchor:
        return None, 0.0, None

    n = len(anchor)
    min_len = max(1, n - MAX_EXTRA)
    max_len = n + MAX_EXTRA

    half = WIDE_WINDOW_WORDS // 2
    search_start = max(0, estimated_cursor - half)
    search_end = min(len(reference_tokens), estimated_cursor + half)

    best_start = best_end = None
    best_score = 0.0

    for start in range(search_start, search_end):
        for span_len in range(min_len, max_len + 1):
            end = start + span_len
            if end > search_end or end > len(reference_tokens):
                continue

            score = difflib.SequenceMatcher(
                None, anchor, reference_tokens[start:end]
            ).ratio()

            if score > best_score:
                best_score, best_start, best_end = score, start, end

    if best_score >= WIDE_FUZZY_THRESHOLD:
        return best_start, best_score, best_end

    return None, best_score, None


def _call_llm_fallback(client, judge_model, segment_text, excerpt):
    prompt = LLM_FALLBACK_PROMPT.format(
        excerpt=excerpt,
        segment=segment_text,
    )

    try:
        r = client.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": 256,
            },
            think=False,
        )

        out = (r.message.content or "").strip()

        if not out or out.upper() == "NONE":
            return None

        seg_wc = segment_word_count(segment_text)
        out_wc = segment_word_count(out)

        if seg_wc > 0 and out_wc > seg_wc * MAX_SPAN_LENGTH_RATIO:
            return None

        return out

    except Exception as e:
        print(f"    ERROR (LLM fallback): {e}")
        return None


def _locate_returned_span(text, reference_tokens):
    """
    Locate LLM-returned span globally after the fact.
    Cheap enough because only successful fallbacks use it.
    """
    target = segment_tokens(text)
    if not target:
        return None

    n = len(target)

    # exact first
    for start in range(0, len(reference_tokens) - n + 1):
        if reference_tokens[start : start + n] == target:
            return start + n

    # small fuzzy fallback
    best_end = None
    best_score = 0.0

    for start in range(0, len(reference_tokens) - n + 1):
        score = difflib.SequenceMatcher(
            None,
            target,
            reference_tokens[start : start + n],
        ).ratio()

        if score > best_score:
            best_score = score
            best_end = start + n

    return best_end if best_score >= 0.85 else None


def _nearest_preceding_anchor(i, anchors):
    for j in range(i - 1, -1, -1):
        if j in anchors:
            return j, anchors[j]
    return -1, 0


def _estimated_cursor(i, segments, anchors, reference_word_count):
    anchor_i, anchor_cursor = _nearest_preceding_anchor(i, anchors)

    advance = sum(segment_word_count(segments[k]) for k in range(anchor_i + 1, i))

    return min(reference_word_count, anchor_cursor + advance)


def _make_excerpt(reference, reference_word_positions, cursor):
    if not reference_word_positions:
        return ""

    w_start = max(0, cursor - LOCAL_WINDOW_WORDS // 2)
    w_end = min(len(reference_word_positions), cursor + LOCAL_WINDOW_WORDS)

    if w_start >= len(reference_word_positions) or w_end <= 0:
        return ""

    return reference[
        reference_word_positions[w_start][0] : reference_word_positions[w_end - 1][1]
    ]


def align_sentence_spans(
    client,
    judge_model,
    reference,
    segments,
    max_workers=DEFAULT_MAX_WORKERS,
):
    reference_tokens, reference_word_positions = tokenize_with_offsets(reference)

    cursor = 0
    spans = {}
    methods = {}
    anchors = {}
    fallback_needed = []

    # ---------------------------------------------------------
    # Pass 1: local mechanical + cheap global exact rescue
    # ---------------------------------------------------------
    for i, segment in enumerate(segments):
        start, score, end = mechanical_match(
            segment,
            reference_tokens,
            cursor,
        )

        if start is not None and score >= MECHANICAL_THRESHOLD:
            spans[i] = span_text(reference, reference_word_positions, start, end)
            methods[i] = "mechanical"
            anchors[i] = end
            cursor = end
            continue

        exact = global_exact_match(segment, reference_tokens)

        if exact is not None:
            start, end = exact
            spans[i] = span_text(reference, reference_word_positions, start, end)
            methods[i] = "global_exact"
            anchors[i] = end
            cursor = end
            continue

        w_start = max(0, cursor - LOCAL_WINDOW_WORDS // 2)
        w_end = min(len(reference_tokens), cursor + LOCAL_WINDOW_WORDS)

        fallback_needed.append((i, segment, w_start, w_end))

        cursor = min(
            len(reference_tokens),
            cursor + segment_word_count(segment),
        )

    # ---------------------------------------------------------
    # Pass 2: first LLM fallback
    # ---------------------------------------------------------
    def _fallback_worker(item):
        _, segment, w_start, w_end = item

        if (
            not reference_word_positions
            or w_start >= len(reference_word_positions)
            or w_end <= 0
        ):
            return None

        excerpt = reference[
            reference_word_positions[w_start][0] : reference_word_positions[w_end - 1][
                1
            ]
        ]

        return _call_llm_fallback(
            client,
            judge_model,
            segment,
            excerpt,
        )

    if fallback_needed:
        results = call_models_in_parallel(
            fallback_needed,
            _fallback_worker,
            max_workers=max_workers,
            progress_every=25,
        )

        for (i, segment, _, _), result in zip(
            fallback_needed,
            results,
        ):
            if result is not None:
                spans[i] = result
                methods[i] = "llm_fallback"

                true_end = _locate_returned_span(
                    result,
                    reference_tokens,
                )
                if true_end is not None:
                    anchors[i] = true_end
            else:
                spans[i] = None
                methods[i] = "unresolved"

    # ---------------------------------------------------------
    # Pass 3: repaired local mechanical + global exact + wide fuzzy
    # ---------------------------------------------------------
    unresolved = [i for i in range(len(segments)) if methods.get(i) == "unresolved"]

    for i in unresolved:
        retry_cursor = _estimated_cursor(
            i,
            segments,
            anchors,
            len(reference_tokens),
        )

        retry_start = max(
            0,
            retry_cursor - max(20, segment_word_count(segments[i])),
        )

        start, score, end = mechanical_match(
            segments[i],
            reference_tokens,
            retry_start,
        )

        if start is not None and score >= MECHANICAL_THRESHOLD:
            spans[i] = span_text(reference, reference_word_positions, start, end)
            methods[i] = "mechanical_repaired"
            anchors[i] = end
            continue

        exact = global_exact_match(
            segments[i],
            reference_tokens,
        )

        if exact is not None:
            start, end = exact
            spans[i] = span_text(reference, reference_word_positions, start, end)
            methods[i] = "global_exact_repaired"
            anchors[i] = end
            continue

        start, score, end = wide_local_fuzzy_match(
            segments[i],
            reference_tokens,
            retry_cursor,
        )

        if start is not None:
            spans[i] = span_text(reference, reference_word_positions, start, end)
            methods[i] = "wide_fuzzy_repaired"
            anchors[i] = end

    # ---------------------------------------------------------
    # Pass 4: second LLM fallback from corrected anchor
    # ---------------------------------------------------------
    still_unresolved = [
        i for i in range(len(segments)) if methods.get(i) == "unresolved"
    ]

    repair_items = []

    for i in still_unresolved:
        retry_cursor = _estimated_cursor(
            i,
            segments,
            anchors,
            len(reference_tokens),
        )

        excerpt = _make_excerpt(
            reference,
            reference_word_positions,
            retry_cursor,
        )

        repair_items.append((i, segments[i], excerpt))

    def _repair_worker(item):
        _, segment, excerpt = item
        if not excerpt:
            return None

        return _call_llm_fallback(
            client,
            judge_model,
            segment,
            excerpt,
        )

    if repair_items:
        results = call_models_in_parallel(
            repair_items,
            _repair_worker,
            max_workers=max_workers,
            progress_every=25,
        )

        for (i, segment, _), result in zip(
            repair_items,
            results,
        ):
            if result is None:
                continue

            spans[i] = result
            methods[i] = "llm_repaired"

            true_end = _locate_returned_span(
                result,
                reference_tokens,
            )
            if true_end is not None:
                anchors[i] = true_end

    stats = {
        "mechanical": sum(m == "mechanical" for m in methods.values()),
        "global_exact": sum(m == "global_exact" for m in methods.values()),
        "mechanical_repaired": sum(
            m == "mechanical_repaired" for m in methods.values()
        ),
        "global_exact_repaired": sum(
            m == "global_exact_repaired" for m in methods.values()
        ),
        "wide_fuzzy_repaired": sum(
            m == "wide_fuzzy_repaired" for m in methods.values()
        ),
        "llm_fallback": sum(m == "llm_fallback" for m in methods.values()),
        "llm_repaired": sum(m == "llm_repaired" for m in methods.values()),
        "unresolved": sum(m == "unresolved" for m in methods.values()),
    }

    return spans, methods, stats
