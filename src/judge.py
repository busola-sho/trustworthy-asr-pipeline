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

IMPORTANT: After any reasoning, your FINAL line must be ONLY the single word true or false."""

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