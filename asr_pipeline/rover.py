"""Word-transition-network voting used by the saved ROVER baseline."""


def align_sequences(first_words: list, second_words: list) -> list:
    """
    Standard edit-distance alignment between two word sequences.
    Returns a list of (word_a_or_None, word_b_or_None) pairs - None marks
    a gap (insertion/deletion) on that side. Match cost 0, substitution
    cost 1, insertion/deletion cost 1. Case-insensitive comparison.
    """
    n, m = len(first_words), len(second_words)
    # dp[i][j] = min cost aligning seq_a[:i] with seq_b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = (
                0 if first_words[i - 1].lower() == second_words[j - 1].lower() else 1
            )
            dp[i][j] = min(
                dp[i - 1][j - 1] + match_cost,  # substitution or match
                dp[i - 1][j] + 1,  # deletion (a has extra word)
                dp[i][j - 1] + 1,  # insertion (b has extra word)
            )

    # backtrace
    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_cost = (
                0 if first_words[i - 1].lower() == second_words[j - 1].lower() else 1
            )
            if dp[i][j] == dp[i - 1][j - 1] + match_cost:
                aligned.append((first_words[i - 1], second_words[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((first_words[i - 1], None))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            aligned.append((None, second_words[j - 1]))
            j -= 1
            continue
        break

    aligned.reverse()
    return aligned


def representative_words(word_voting_columns: list, model_order: list) -> list:
    """
    Builds the sequence used to align the NEXT system against - one word
    per existing column, using the current majority word so far (or the
    first available system's word if no majority yet). This is what the
    new hypothesis gets aligned against, per Fiscus's incremental merge.
    """
    representative_sequence = []
    for column in word_voting_columns:
        words = [column.get(m) for m in model_order if column.get(m) is not None]
        if not words:
            representative_sequence.append("")
            continue
        # majority so far, ties broken by first-seen order
        counts = {}
        for w in words:
            counts[w.lower()] = counts.get(w.lower(), 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        # recover original casing from the first matching word
        representative_sequence.append(next(w for w in words if w.lower() == best))
    return representative_sequence


def build_word_voting_columns(hypotheses: dict, model_order: list) -> list:
    """
    Incrementally builds the Word Transition Network by aligning each
    system's hypothesis into the growing composite, one at a time.
    Returns a list of columns, each a dict {model_name: word_or_None}.
    """
    first_model = model_order[0]
    first_words = hypotheses[first_model].split()
    word_voting_columns = [{first_model: w} for w in first_words]

    for model in model_order[1:]:
        new_words = hypotheses[model].split()
        representative_sequence = representative_words(word_voting_columns, model_order)
        alignment = align_sequences(representative_sequence, new_words)

        merged = []
        column_index = 0
        for rep_word, new_word in alignment:
            if rep_word is not None and new_word is not None:
                # aligned to an existing column - extend it
                column = dict(word_voting_columns[column_index])
                column[model] = new_word
                merged.append(column)
                column_index += 1
            elif rep_word is not None and new_word is None:
                # existing column has no match in the new hypothesis - deletion for this model
                column = dict(word_voting_columns[column_index])
                column[model] = None
                merged.append(column)
                column_index += 1
            else:
                # new hypothesis has an extra word not in the existing WTN - new column,
                # None for every system that came before this one
                column = {m: None for m in model_order if m != model}
                column[model] = new_word
                merged.append(column)

        word_voting_columns = merged

    # fill in any missing models per column with None (systems that never
    # got a chance to vote at that column)
    for column in word_voting_columns:
        for m in model_order:
            if m not in column:
                column[m] = None

    return word_voting_columns


def get_word_confidence(model: str, word: str, segments: list) -> float:
    """Looks up the confidence score for a specific word from a model's
    segments list. Falls back to 0.5 if not found (shouldn't normally
    happen, but keeps voting well-defined if segment data is sparse)."""
    if not segments:
        return 0.5
    for segment in segments:
        if segment.get("word", "").lower() == word.lower():
            return segment.get("confidence", 0.5) or 0.5
    return 0.5


def vote_column(column: dict, model_order: list, segments_by_model: dict) -> str:
    """
    Majority vote for one WTN column. NULL (None) is a valid candidate,
    representing "this word shouldn't be here". Ties are broken by
    average confidence of the tied words' proposing systems, per Fiscus.
    Returns the winning word, or None if NULL wins (word is dropped).
    """
    votes = {}  # word_or_None -> list of models that proposed it
    for model in model_order:
        word = column.get(model)
        votes.setdefault(word, []).append(model)

    # majority = most votes
    max_votes = max(len(asr_models) for asr_models in votes.values())
    tied = [word for word, asr_models in votes.items() if len(asr_models) == max_votes]

    if len(tied) == 1:
        return tied[0]

    # tiebreak: average confidence of the proposing systems for each tied candidate
    best_word, best_conf = None, -1.0
    for word in tied:
        asr_models = votes[word]
        if word is None:
            # NULL's "confidence" - treat as the average of NOT proposing,
            # use a neutral 0.5 baseline so NULL isn't unfairly favoured/disfavoured
            conf = 0.5
        else:
            confs = [
                get_word_confidence(m, word, segments_by_model.get(m, []))
                for m in asr_models
            ]
            conf = sum(confs) / len(confs) if confs else 0.5
        if conf > best_conf:
            best_word, best_conf = word, conf

    return best_word


def rover_combine(hypotheses: dict, model_order: list, segments_by_model: dict) -> str:
    """Full ROVER pipeline for one sample: build WTN, vote per column,
    concatenate winning words."""
    word_voting_columns = build_word_voting_columns(hypotheses, model_order)
    winners = [
        vote_column(col, model_order, segments_by_model) for col in word_voting_columns
    ]
    return " ".join(w for w in winners if w)
