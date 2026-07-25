"""
scripts/finetuning/chunking.py

Splits long audio samples into <=MAX_CHUNK_SEC segments with correctly
aligned transcript text, for feeding into Whisper (which truncates audio
to ~30s while still silently accepting a full, longer transcript as the
target - a corrupted training pair if left unfixed).

WHY THIS MATTERS SPECIFICALLY FOR YOUR DATA: CommonVoice averages ~55
seconds per sample (10.45h / 680 samples) - well over Whisper's 30s
window. English Dialects (~5.2s avg) and EdAcc (~3.8s avg) are mostly
fine already, but any dataset could have occasional long outliers, so
this chunker runs generically over all three (a sample under
MAX_CHUNK_SEC is returned unchanged, no chunking overhead).

HOW ALIGNMENT WORKS (since raw dataset audio has no ground-truth
per-word timestamps):
  1. Look up the SAME sample_index in the WhisperX canonical benchmark
     file - it already has real, forced-aligned word-level timestamps,
     just for WhisperX's own hypothesis text, not the reference.
  2. Align reference words to WhisperX's hypothesis words via the same
     edit-distance DP alignment rover.py already uses for voting
     (case-insensitive word matching).
  3. Transfer each aligned hypothesis word's (start, end) onto the
     matching reference word. Reference words that didn't align to any
     hypothesis word (insertions relative to WhisperX's hyp) get a
     timestamp interpolated between their nearest aligned neighbors.
  4. Walk through timestamped reference words, cutting a new chunk
     whenever adding the next word would push the running chunk past
     MAX_CHUNK_SEC, splitting audio at the midpoint between the last
     word of one chunk and the first word of the next.

FALLBACK: if WhisperX has no entry for this sample_index (missing data,
mismatched indexing, etc.), falls back to an even time-proportional
split across words - noted via a printed warning, since it's a weaker
approximation than the real alignment path.
"""

import numpy as np

from src.selector import find_canonical_file, load_samples

MAX_CHUNK_SEC = 28.0   # a little under Whisper's 30s ceiling, for safety margin
MIN_CHUNK_SEC = 0.3    # drop pathologically short leftover chunks (likely noise)

_whisperx_cache = {}


def _get_whisperx_sample(dataset: str, sample_index: int):
    if dataset not in _whisperx_cache:
        try:
            path = find_canonical_file("whisperx", dataset)
            samples = load_samples(path)
            _whisperx_cache[dataset] = {
                s["sample_index"]: s for s in samples if s.get("sample_index") is not None
            }
        except FileNotFoundError:
            _whisperx_cache[dataset] = {}
    return _whisperx_cache[dataset].get(sample_index)


def _align_words(ref_words: list, hyp_words: list) -> list:
    """
    Standard edit-distance word alignment (same approach as
    rover.py's align_sequences) - returns a list of (ref_word_or_None,
    hyp_idx_or_None) pairs. hyp_idx (not the word itself) is returned so
    the caller can look up that word's timestamp directly.
    """
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1].lower() == hyp_words[j - 1].lower() else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((i - 1, None))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned.append((None, j - 1))
            j -= 1
            continue
        break
    aligned.reverse()
    return aligned


def _get_ref_word_timestamps(dataset: str, sample_index: int, ref_text: str, total_duration: float):
    """
    Returns a list of (word, start, end) for every word in ref_text, in
    order. Uses WhisperX's real timestamps where alignment succeeds,
    interpolates for unaligned reference words. Returns None if no
    WhisperX data is available for this sample (caller should fall back
    to an even proportional split).
    """
    ref_words = ref_text.split()
    if not ref_words:
        return []

    whisperx_sample = _get_whisperx_sample(dataset, sample_index)
    segments = (whisperx_sample or {}).get("segments")
    if not segments:
        return None

    hyp_words = [seg["word"] for seg in segments]
    hyp_starts = [seg.get("start") for seg in segments]
    hyp_ends = [seg.get("end") for seg in segments]

    if not hyp_words or any(s is None for s in hyp_starts) or any(e is None for e in hyp_ends):
        return None

    alignment = _align_words(ref_words, hyp_words)

    # first pass: assign known timestamps from aligned hyp words
    ref_timestamps = [None] * len(ref_words)
    for ref_idx, hyp_idx in alignment:
        if ref_idx is not None and hyp_idx is not None:
            ref_timestamps[ref_idx] = (hyp_starts[hyp_idx], hyp_ends[hyp_idx])

    # second pass: interpolate any ref words that didn't align to a hyp word
    known_indices = [i for i, ts in enumerate(ref_timestamps) if ts is not None]
    if not known_indices:
        return None  # nothing usable - caller falls back to proportional split

    for i in range(len(ref_timestamps)):
        if ref_timestamps[i] is not None:
            continue
        # find nearest known neighbors before/after
        before = [k for k in known_indices if k < i]
        after = [k for k in known_indices if k > i]
        if before and after:
            b, a = before[-1], after[0]
            b_end = ref_timestamps[b][1]
            a_start = ref_timestamps[a][0]
            ref_timestamps[i] = (b_end, a_start) if a_start > b_end else (b_end, b_end)
        elif before:
            b_end = ref_timestamps[before[-1]][1]
            ref_timestamps[i] = (b_end, min(b_end + 0.3, total_duration))
        elif after:
            a_start = ref_timestamps[after[0]][0]
            ref_timestamps[i] = (max(a_start - 0.3, 0.0), a_start)

    return [(w, ts[0], ts[1]) for w, ts in zip(ref_words, ref_timestamps)]


def _proportional_fallback_timestamps(ref_text: str, total_duration: float):
    """Even time-proportional split across words - used only when no
    WhisperX alignment data is available. Weaker approximation, printed
    as a warning by the caller."""
    ref_words = ref_text.split()
    if not ref_words:
        return []
    per_word = total_duration / len(ref_words)
    return [(w, i * per_word, (i + 1) * per_word) for i, w in enumerate(ref_words)]


def chunk_sample(dataset: str, sample_index: int, ref_text: str, audio: np.ndarray, sample_rate: int):
    """
    Returns a list of (chunk_text, chunk_audio) tuples. If the sample is
    already under MAX_CHUNK_SEC, returns a single-item list with the
    original text/audio unchanged (no chunking overhead).
    """
    total_duration = len(audio) / sample_rate

    if total_duration <= MAX_CHUNK_SEC:
        return [(ref_text, audio)]

    word_timestamps = _get_ref_word_timestamps(dataset, sample_index, ref_text, total_duration)
    if word_timestamps is None:
        print(f"    WARNING: no WhisperX alignment for {dataset}/{sample_index} "
              f"({total_duration:.1f}s) - using proportional fallback split (weaker approximation)")
        word_timestamps = _proportional_fallback_timestamps(ref_text, total_duration)

    if not word_timestamps:
        return [(ref_text, audio)]

    chunks = []
    current_words = []
    current_start = word_timestamps[0][1]

    for idx, (word, start, end) in enumerate(word_timestamps):
        # would adding this word push the chunk over the limit?
        if current_words and (end - current_start) > MAX_CHUNK_SEC:
            # close out the current chunk at the midpoint between the
            # last included word's end and this word's start
            prev_end = current_words[-1][2]
            boundary = (prev_end + start) / 2 if start > prev_end else prev_end
            chunks.append((current_words, current_start, boundary))
            current_words = []
            current_start = boundary

        current_words.append((word, start, end))

    if current_words:
        chunks.append((current_words, current_start, total_duration))

    results = []
    for words, chunk_start, chunk_end in chunks:
        chunk_start = max(0.0, min(chunk_start, total_duration))
        chunk_end = max(chunk_start, min(chunk_end, total_duration))
        if (chunk_end - chunk_start) < MIN_CHUNK_SEC:
            continue
        start_sample = int(chunk_start * sample_rate)
        end_sample = int(chunk_end * sample_rate)
        chunk_audio = audio[start_sample:end_sample]
        chunk_text = " ".join(w for w, _, _ in words)
        if chunk_text.strip() and len(chunk_audio) > 0:
            results.append((chunk_text, chunk_audio))

    return results if results else [(ref_text, audio)]