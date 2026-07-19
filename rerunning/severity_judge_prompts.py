"""
Severity scoring prompts for ASR transcription error evaluation.
Two conditions for comparison: DIRECT_SEVERITY_PROMPT vs STRUCTURED_SEVERITY_PROMPT.
Both score against the same 0-4 rubric; the structured condition forces
error-type identification and consequence reasoning before the final score.
"""

SEVERITY_RUBRIC = """
Severity scale (0-4), describing the meaning-level impact of ASR transcription
errors on a reference (ground-truth) vs hypothesis (ASR output) sentence pair.
Judge only from the pair given - do not assume outside context.

0 - No meaning change: surface form only (casing, punctuation, filler words,
    dialectal/phonetic variation, or an unambiguous spelling variant of the
    same proper noun) - no change to propositional content.
1 - Trivial/cosmetic error: wording differs or is garbled, but the meaning
    is confidently and cleanly recoverable, no material fact changes. Covers
    nonsensical renderings with no real alternative (e.g. "Japer" for
    "Jaipur") - even where the substituted words are real, if the resulting
    phrase doesn't cohere into a believable alternative claim, treat it the
    same way (e.g. "seeing her goggle" for "saying hey Google"). Requires
    CONFIDENT, CLEAN recovery: if reconstruction is needed, or any doubt
    remains, use level 2 instead. For longer, multi-clause utterances,
    prefer 2 over 1 for any genuine (even minor) content-word change - more
    text means more room for a subtle shift to go unnoticed.
2 - Ambiguous meaning shift: genuine ambiguity, real reconstruction needed,
    or residual doubt about intent. Covers connector/relational word changes
    (and/or/in/of/on) that alter how two items relate rather than what they
    individually are (e.g. "activities ON work time" -> "activities AND work
    time" turns one benefit into two vague ones); the same word garbled
    differently more than once in a sentence; and total, unrecoverable loss
    of a detail (content simply gone, not just unclear) - loss alone stays
    at 2, never 3/4, since those levels require a CLEAR meaning change to be
    asserted, and severe garbling can't assert anything clearly. No
    unambiguous different material fact is asserted at this level.
3 - Clear factual error: unambiguously asserts a different material fact -
    entity, number, time, location, quantity, or action - with no real
    ambiguity about the new claim. Includes substitution of a real,
    plausible, different entity (e.g. "Joan" for "John", "Jasper" for
    "Jaipur", "Shannon" for "Schengen"), as opposed to nonsense with no real
    alternative (level 1).
4 - Critical meaning alteration: reverses, fabricates, or fundamentally
    changes the central assertion - flipped negation, a tense/aspect shift
    that changes whether a consequential action happened/will happen, a
    different actor, a reversed alibi/sequence, or fabricated content. Does
    NOT require a literal negation word: any swap (demonstrative, pronoun,
    modal, tense) that flips the claim with nothing in the sentence to
    correct it qualifies (e.g. "This is not a better way" -> "There is not
    a better way" reverses the recommendation). Also covers hallucinated
    content filling a span marked inaudible/unclear (e.g. [inc]) - score
    as 4 by default, since there's no ground truth to check it against.
"""

ERROR_TYPE_TAXONOMY = """
Error type taxonomy - identify the PRIMARY error type that contributes most
to the final severity. Other errors may be mentioned in the justification.
- Acceptable variation (typical severity: 0): dialectal/phonetic spelling,
  casing, punctuation, filler words - no content change.
- Surface-form error (typical severity: 0-1): wording or minor spelling
  differences that do not change the underlying claim.
- Content substitution (typical severity: 1-4): a CONTENT-BEARING word or
  phrase is replaced - this means entities, numbers, times, locations,
  quantities, objects, or actions. Do NOT use this category for connector
  or relational words (and/or/in/of/on) - those belong under "Addition,
  omission, or structural error" below, even though they are also
  substitutions in a loose sense. Use the rubric and consequence criteria
  to determine whether the replacement is harmless, ambiguous, factual, or
  critical.
- Addition, omission, or structural error (typical severity: 1-4): words,
  clauses, or claims are added, dropped, or reordered in a way that may
  change what is asserted. This INCLUDES connector/relational word changes
  (and/or/in/of/on) that change how two things relate to each other, not
  just what they individually are.
- Polarity or modality error (typical severity: 1-4): negation, certainty,
  modality, or TENSE/ASPECT is altered (e.g. "did" vs "did not", "may have"
  vs "did", present vs past tense changing whether something is ongoing,
  completed, or hadn't happened yet). Also covers claim reversals achieved
  WITHOUT a literal negation word - e.g. a demonstrative or pronoun swap
  ("this" vs "there") that flips what is actually being claimed. WATCH FOR:
  a dialect word normalized to standard English in the WRONG tense (e.g.
  "dinnae" [present: don't] rendered as "didn't" [past]) looks like the
  free dialect normalization in level 0, but is not - the tense changed,
  not just the spelling convention. Severity for tense/aspect shifts scales
  with materiality and correction: 1-2 if isolated and the surrounding
  context still supports the correct reading; 4 if it changes whether a
  consequential action happened/will happen with nothing to correct it.
"""

DIRECT_SEVERITY_PROMPT = """You are evaluating the severity of a speech-to-text
transcription error for use in a high-stakes policing context, where the
transcript may be relied on as evidence.

{rubric}

Reference (ground truth): "{reference}"
Hypothesis (ASR output): "{hypothesis}"

Compare the reference and hypothesis, and assign a single severity score from
0 to 4 using the scale above. Base your judgement only on the pair given -
do not assume outside context.

Respond in exactly this format:
Severity: <0-4>
Justification: <one to two sentences explaining your score>
""".format(rubric=SEVERITY_RUBRIC, reference="{reference}", hypothesis="{hypothesis}")


STRUCTURED_SEVERITY_PROMPT = """You are evaluating the severity of a speech-to-text
transcription error for use in a high-stakes policing context, where the
transcript may be relied on as evidence.

{rubric}

{error_types}

Consequence criteria - after identifying the error type, reason through these
four questions to decide where within that type's range the severity falls.
Do not score these individually; use them to inform a single holistic score.
1. Materiality: does the error alter information important to understanding
   the incident - identity, action, object, time, location, quantity, intent,
   or sequence of events?
2. Directionality: does it merely change how something is expressed, or does
   it change the underlying claim about what happened, who did it, or
   whether it happened at all?
3. Recoverability: can the intended meaning be confidently recovered from the
   pair alone? Reconstruction or residual doubt is a reason to score higher,
   not a reason to round down.
4. Misleading plausibility: is the erroneous transcript fluent enough to be
   accepted as true, or obviously malformed and likely to be questioned? A
   fluent but wrong transcript is more dangerous than a broken one, since it
   creates false confidence rather than scrutiny. Apply the real-vs-
   nonsensical distinction from levels 1 and 3 above. Note total,
   unrecoverable loss (see level 2) scores low here even though
   recoverability is at its worst - broken garbling isn't a believable claim.

Reference (ground truth): "{reference}"
Hypothesis (ASR output): "{hypothesis}"

Respond in exactly this format:
Error type (primary): <one of the five types above>
Materiality: <brief reasoning>
Directionality: <brief reasoning>
Recoverability: <brief reasoning>
Misleading plausibility: <brief reasoning>
Severity: <0-4>
Justification: <one to two sentences tying the reasoning above to the final score>
""".format(rubric=SEVERITY_RUBRIC, error_types=ERROR_TYPE_TAXONOMY,
           reference="{reference}", hypothesis="{hypothesis}")