"""
src/judge.py

Shared LLM judge utilities for Meaning Alteration Rate (MAR) scoring
and severity-scaled scoring. Centralises prompts, normalisation, and
Ollama calls so they are not copy-pasted across selector scripts.

Usage:
    from src.judge import ollama_mar, ollama_severity, normalise, parse_verdict
"""

import re
import time
import torch
from ollama import Client

OLLAMA_HOST  = "http://localhost:11434"
JUDGE_MODEL  = "qwen2.5:7b"

# ── MAR prompt (v2 judge — named entities excluded) ────────────────────────────

MAR_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause someone to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

NAMED ENTITIES: Ignore spelling or form variation in people's names, place names, and organisation names (e.g. "Forfar" vs "Forfa", "Mhairi" vs "Maria"). Even human transcribers cannot reliably spell unfamiliar names from audio alone. Only flag a named entity as an error if it is unambiguously a DIFFERENT entity altogether (wrong city, wrong person, a number that changes which date/amount is referenced) — not merely a different spelling of the same intended entity.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number refers to a genuinely different entity (not just a spelling variant)
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Example:
Reference: My neighbour Mhairi McTaggart said she heard the noise around midnight.
Hypothesis: My neighbour Maria MacTaggart said she heard the noise around midnight.
This refers to the same person and the same claim — the spelling differs but no meaning changed.
Answer: false

IMPORTANT: Reply with ONLY the single word true or false. No explanation, no reasoning, no other text."""

# ── Severity prompt (0-4 scale, named entities excluded) ──────────────────────

SEVERITY_PROMPT = """You are evaluating ASR transcripts in a high-stakes context where accuracy matters. Given a reference and hypothesis transcript of the same audio, rate the SEVERITY of any meaning-altering error on a scale of 0-4.

IMPORTANT — NAMED ENTITIES: Ignore spelling or form variation in people's names, place names, and organisation names (e.g. "Forfar" vs "Forfa", "Mhairi" vs "Maria"). Even human transcribers cannot reliably spell unfamiliar names from audio alone. Only score a named entity as an error if it is unambiguously a DIFFERENT entity altogether (wrong city, wrong person, wrong date/amount) — not merely a different spelling of the same intended entity.

SCALE:
0 = No meaning change. Either a perfect match, or differences are purely surface-level: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"->"didn't", "oot"->"out"), filler words, or named entity spelling variation (see above).
1 = Trivial/cosmetic error. A word changed but does not affect practical interpretation (e.g. "Rd" vs "Road", a digit vs spelled-out number, a minor synonym swap with no factual consequence).
2 = Ambiguous or softened meaning shift. Something changed that could matter depending on context but is not a clean factual reversal (e.g. hedging language altered such as "might have" becoming "did", a vague pronoun reference changed, a shift in certainty or tone).
3 = Clear factual error on a non-critical detail. A wrong name, place, number, or date that is incorrect but does not flip the substance of the account (e.g. wrong street name in an otherwise correctly identified area, a time off by a small margin).
4 = Critical meaning reversal or fabrication. Negation added or removed, an alibi reversed, content hallucinated over an inaudible segment, or a name/place/number error that changes who/where/when in a way that materially affects the account.

Examples:
Reference: She said she wisnae near the pub on Saturday night.
Hypothesis: She said she was near the pub on Saturday night.
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: 4

Reference: He works the back shift at the factory on Keppoch Road.
Hypothesis: He works the back shift at the factory on Keppoch Rd.
Reasoning: "Rd" is a standard abbreviation for Road; same location, no factual content lost.
Answer: 1

Reference: I think it was around three or four people at the meeting.
Hypothesis: There were three or four people at the meeting.
Reasoning: Hedging ("I think... around") dropped, making an uncertain estimate sound definite.
Answer: 2

Reference: He said he saw her near Gorbals Street that evening.
Hypothesis: He said he saw her near Garscube Street that evening.
Reasoning: Wrong street name but core claim unchanged.
Answer: 3

Reference: My neighbour Mhairi McTaggart said she heard the noise around midnight.
Hypothesis: My neighbour Maria MacTaggart said she heard the noise around midnight.
Reasoning: Same person, same claim, spelling differs only. Named entity variation excluded.
Answer: 0

IMPORTANT: Reply with ONLY the single digit 0, 1, 2, 3, or 4. No explanation, no reasoning, no other text."""


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase and strip punctuation for WER computation."""
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Verdict parsing ────────────────────────────────────────────────────────────

def parse_verdict(result: str):
    """Parse true/false from LLM response, handling reasoning preamble."""
    result = result.strip().lower()
    if result.startswith("true"):
        return True
    elif result.startswith("false"):
        return False
    elif "true" in result and "false" not in result:
        return True
    elif "false" in result and "true" not in result:
        return False
    return None


def parse_severity(result: str):
    """Parse 0-4 severity score from LLM response."""
    result = result.strip()
    match = re.search(r'[0-4]', result)
    if match:
        return int(match.group())
    return None


# ── Ollama judge calls ─────────────────────────────────────────────────────────

def get_client(host: str = OLLAMA_HOST) -> Client:
    return Client(host=host)


def ollama_mar(
    client: Client,
    ref: str,
    hyp: str,
    sample_wer_val: float,
    model: str = JUDGE_MODEL,
    sleep: float = 0.05,
) -> bool:
    """
    Binary MAR verdict. Returns False immediately if WER is 0 (no errors).
    Returns None on failure.
    """
    if sample_wer_val == 0:
        return False
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": MAR_PROMPT},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
            options={"temperature": 0},
        )
        time.sleep(sleep)
        return parse_verdict(response.message.content)
    except Exception as e:
        print(f"  ERROR (MAR judge): {e}")
        return None


def ollama_severity(
    client: Client,
    ref: str,
    hyp: str,
    model: str = JUDGE_MODEL,
    retries: int = 2,
    sleep: float = 0.05,
):
    """
    0-4 severity score. Returns None on failure after retries.
    """
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SEVERITY_PROMPT},
                    {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
                ],
                options={"temperature": 0},
            )
            severity = parse_severity(response.message.content)
            if severity is not None:
                time.sleep(sleep)
                return severity
            print(f"  WARNING: could not parse severity: '{response.message.content[:80]}'")
        except Exception as e:
            print(f"  ERROR (severity judge, attempt {attempt+1}): {e}")
        time.sleep(0.2)
    return None


def check_model_available(client: Client, model: str = JUDGE_MODEL) -> bool:
    """Check whether a given Ollama model is pulled and available."""
    try:
        available = [m.model for m in client.list().models]
        return any(model in m for m in available)
    except Exception:
        return False

# This is the new function to add to src/judge.py after ollama_mar()

SCOTTISH_ANNOTATIONS = {
    "couldnae": "could not", "cannae": "cannot", "cannae": "cannot",
    "wisnae": "wasn't", "wasnae": "wasn't", "isnae": "isn't",
    "dinnae": "don't", "didnae": "didn't", "wouldnae": "wouldn't",
    "shouldnae": "shouldn't", "hasnae": "hasn't", "havnae": "haven't",
    "willnae": "won't", "arenae": "aren't", "werenae": "weren't",
    "nae": "no/not", "naw": "no",
    "oot": "out", "tae": "to", "fae": "from",
    "wee": "small/little", "aye": "yes", "ken": "know",
    "braw": "good", "doon": "down", "hoose": "house",
    "wi": "with", "mair": "more", "hae": "have",
    "dae": "do", "gie": "give", "tak": "take",
}


def annotate_scottish(text: str) -> str:
    """Annotate Scottish dialect words inline with their standard English meaning."""
    import re
    words = re.split(r'(\s+)', text)
    result = []
    for token in words:
        clean = token.lower().strip(".,!?;:'\"")
        if clean in SCOTTISH_ANNOTATIONS:
            result.append(f"{token} [={SCOTTISH_ANNOTATIONS[clean]}]")
        else:
            result.append(token)
    return "".join(result)


SENTENCE_MAR_PROMPT = """You are evaluating one sentence from an ASR hypothesis transcript of a Scottish English police interview.

The hypothesis transcript is a full transcription of the same audio as the reference transcript.
The sentence you are evaluating is ONE PART of that full hypothesis transcript.

FULL REFERENCE TRANSCRIPT (Scottish dialect words annotated with their standard English meaning in [=...]):
{ref_annotated}

ONE SENTENCE FROM THE HYPOTHESIS TRANSCRIPT:
{hyp_sentence}

Your task:
1. Find the part of the reference transcript that this hypothesis sentence is trying to transcribe
2. Check whether the CORE FACTUAL MEANING has been altered

Answer TRUE if the meaning changed — e.g. a negation was lost, a wrong name/place/number was used.
Answer FALSE if the meaning is preserved.

The following are NOT errors — answer FALSE if the only differences are:
- Dialect normalisation: wisnae→wasn't, couldnae→could not, oot→out, wee→small etc.
- Filler words removed: um, uh, eh, you know, I mean, like, right
- False starts removed: words ending in - (e.g. "gr-", "th-", "s-")
- Repetitions reduced: "so so wrong" → "so wrong"
- Cleaner phrasing of the same content
- Named entity spelling variation (Mhairi→Maria)
- The hypothesis sentence is shorter than the reference — content may appear in other sentences

Answer TRUE ONLY if a core factual claim is wrong — wrong negation, wrong name/place/number, fabricated content.

Reply with ONLY the single word: true or false"""


def ollama_sentence_mar(
    client,
    ref_full: str,
    hyp_sentence: str,
    model: str = JUDGE_MODEL,
    sleep: float = 0.05,
) -> bool:
    """
    Sentence-level MAR verdict.

    Evaluates one hypothesis sentence against the full reference transcript.
    Scottish dialect words in the reference are annotated with their standard
    English meaning to help the judge recognise dialect normalisation as correct.

    Args:
        client:        Ollama client
        ref_full:      full reference transcript (ground truth)
        hyp_sentence:  single hypothesis sentence to evaluate
        model:         Ollama model name
        sleep:         sleep after call

    Returns:
        True  = meaning-altering error in this sentence
        False = sentence is accurate
        None  = judge call failed
    """
    if not hyp_sentence or not hyp_sentence.strip():
        return False

    ref_annotated = annotate_scottish(ref_full)

    prompt = SENTENCE_MAR_PROMPT.format(
        ref_annotated=ref_annotated,
        hyp_sentence=hyp_sentence,
    )

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        time.sleep(sleep)
        return parse_verdict(response.message.content)
    except Exception as e:
        print(f"  ERROR (sentence MAR judge): {e}")
        return None

SENTENCE_SEVERITY_PROMPT = """You are evaluating one sentence from an ASR hypothesis transcript of a Scottish English police interview.

The hypothesis transcript is a full transcription of the same audio as the reference transcript.
The sentence you are evaluating is ONE PART of that full hypothesis transcript.

FULL REFERENCE TRANSCRIPT (Scottish dialect words annotated with their standard English meaning in [=...]):
{ref_annotated}

ONE SENTENCE FROM THE HYPOTHESIS TRANSCRIPT:
{hyp_sentence}

Your task:
1. Find the part of the reference transcript that this hypothesis sentence is trying to transcribe
2. Rate the SEVERITY of any meaning-altering error on a scale of 0-4

SCALE:
0 = Meaning preserved. Only surface-level differences, such as filler removal, dialect normalisation, cleaner phrasing, or harmless spelling variation.
1 = Negligible error. A minor difference is present, but the intended meaning remains clear.
2 = Moderate ambiguity. Meaning has become less certain or slightly shifted, but can still be recovered from context.
3 = Major factual distortion. A central fact changes, such as the person, place, number, date, action, timing, or responsibility.
4 = Critical contradiction or fabrication. A consequential claim is reversed, removed, or invented, such as lost negation, reversed responsibility, or hallucinated content.

The hypothesis sentence is always shorter than the full reference — this is expected.

Reply with ONLY the single digit: 0, 1, 2, 3, or 4"""


def ollama_sentence_severity(
    client,
    ref_full: str,
    hyp_sentence: str,
    model: str = JUDGE_MODEL,
    sleep: float = 0.05,
):
    if not hyp_sentence or not hyp_sentence.strip():
        return 0

    ref_annotated = annotate_scottish(ref_full)

    prompt = SENTENCE_SEVERITY_PROMPT.format(
        ref_annotated=ref_annotated,
        hyp_sentence=hyp_sentence,
    )

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        time.sleep(sleep)
        return parse_severity(response.message.content)
    except Exception as e:
        print(f"  ERROR (sentence severity judge): {e}")
        return None