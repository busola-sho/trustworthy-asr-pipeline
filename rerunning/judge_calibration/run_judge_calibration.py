"""
rerunning/judge_calibration/run_judge_calibration.py

Runs all judges (5 local via Ollama, 2 via API) across both the
DIRECT and STRUCTURED severity prompt conditions, against the human-annotated
candidate_pool.json. Writes one result file per (judge, condition), resumable.

Judges:
    qwen3.5     - local, Ollama
    gemma4      - local, Ollama
    phi4        - local, Ollama (confirmed tag: phi4:14b)
    ministral3  - local, Ollama (confirmed tag: ministral-3:14b, note hyphen)
    qwen2.5     - local, Ollama (optional 6th/legacy comparison judge)
    deepseek    - API, DeepSeek V4 Flash (model string: deepseek-v4-flash)
    gpt5.6luna  - API, GPT-5.6 Luna (model string: gpt-5.6-luna) - cheap
                  second API comparison point, needs OPENAI_API_KEY

Confirm qwen3.5 and gemma4 tags with `ollama list` before running - phi4 and
ministral3 tags below are already confirmed working.

Usage:
    python rerunning/judge_calibration/run_judge_calibration.py \\
        --judge qwen3.5 --condition direct
    python rerunning/judge_calibration/run_judge_calibration.py \\
        --judge all --condition all
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client
from openai import OpenAI

from severity_judge_prompts import DIRECT_SEVERITY_PROMPT, STRUCTURED_SEVERITY_PROMPT, MEDIUM_SEVERITY_PROMPT

load_dotenv()  # picks up DEEPSEEK_API_KEY from a .env file in the working directory

OLLAMA_HOST = "http://localhost:11434"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash"
GPT56_MODEL = "gpt-5.6-luna"

CANDIDATE_POOL_PATH = "writeup_results/candidate_pool.json"
RESULTS_DIR = "writeup_results/judge_calibration"

# --- qwen2.5 is an OPTIONAL 6th judge - your original/legacy selector judge,
# already pulled, included for comparison against the new lineup. Not part
# of the core 5, but runnable the same way via --judge qwen2.5. ---
LOCAL_JUDGES = {
    "qwen3.5":    "qwen3.5:9b",
    "gemma4":     "gemma4:12b",
    "phi4":       "phi4:14b",
    "ministral3": "ministral-3:14b",
    "qwen2.5":    "qwen2.5:7b",
}

# Each API judge maps to (model_name, base_url, env_var_for_api_key, temperature).
# base_url=None means use OpenAI's default endpoint.
# temperature=None means omit the parameter entirely (some models, like
# GPT-5.6 Luna, only support their default temperature and error on override).
API_JUDGES = {
    "deepseek":   (DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, "DEEPSEEK_API_KEY", 0),
    "gpt5.6luna": (GPT56_MODEL, None, "OPENAI_API_KEY", None),
}

ALL_JUDGES = list(LOCAL_JUDGES.keys()) + list(API_JUDGES.keys())

PROMPTS = {
    "direct":     DIRECT_SEVERITY_PROMPT,
    "medium":     MEDIUM_SEVERITY_PROMPT,
    "structured": STRUCTURED_SEVERITY_PROMPT,
}


def parse_severity_response(text: str):
    """
    Extract the final severity digit from a judge response, whether it came
    from the direct condition (just 'Severity: X') or the structured
    condition (multi-field breakdown ending in 'Severity: X'). Looks
    specifically for a 'Severity:' label rather than the first digit in the
    text, since structured responses mention other numbers in their
    reasoning (e.g. referencing rubric levels) before the final answer.
    """
    match = re.search(r"severity\s*:\s*([0-4])", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # fallback: last standalone 0-4 digit in the text
    fallback = re.findall(r"(?<!\d)[0-4](?!\d)", text)
    return int(fallback[-1]) if fallback else None


def call_local_judge(client: Client, model_tag: str, prompt: str, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model_tag,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
                think=False,  # disables reasoning-mode output for models that
                              # support it (gemma4, qwen3.5, etc) - avoids the
                              # thinking trace eating the token budget before
                              # reaching the final "Severity:" line
            )
            text = response.message.content
            severity = parse_severity_response(text)
            if severity is not None:
                return severity, text, None
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            if attempt == retries:
                return None, None, str(e)
            time.sleep(1.0)
    return None, text, "could_not_parse_severity"


def call_api_judge(client: OpenAI, model_name: str, prompt: str,
                    temperature=0, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            severity = parse_severity_response(text)
            if severity is not None:
                return severity, text, None
            if attempt < retries:
                time.sleep(0.5)
        except Exception as e:
            if attempt == retries:
                return None, None, str(e)
            time.sleep(1.0)
    return None, text, "could_not_parse_severity"


def load_candidate_pool(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def run_judge_condition(judge_name: str, condition: str, pool: list, output_path: str):
    prompt_template = PROMPTS[condition]

    # resume support: load existing results, skip uids already done
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_uids = {r["uid"] for r in results if r.get("severity") is not None}
    else:
        results = []
        done_uids = set()

    is_local = judge_name in LOCAL_JUDGES
    if is_local:
        client = Client(host=OLLAMA_HOST)
        model_tag = LOCAL_JUDGES[judge_name]
    else:
        model_name, base_url, env_var, temperature = API_JUDGES[judge_name]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise SystemExit(
                f"{env_var} environment variable not set (needed for {judge_name}). "
                f"Set it in your .env file or with: export {env_var}=your_key_here"
            )
        client = OpenAI(base_url=base_url, api_key=api_key) if base_url \
            else OpenAI(api_key=api_key)

    print(f"\n── {judge_name} / {condition} ──")
    n_total = len(pool)
    n_skip = sum(1 for s in pool if s["uid"] in done_uids)
    print(f"  {n_total} sentences, {n_skip} already done, resuming...")

    for i, sample in enumerate(pool):
        uid = sample["uid"]
        if uid in done_uids:
            continue

        prompt = prompt_template.format(
            reference=sample["ref"],
            hypothesis=sample["hyp"],
        )

        if is_local:
            severity, raw_response, error = call_local_judge(client, model_tag, prompt)
        else:
            severity, raw_response, error = call_api_judge(
                client, model_name, prompt, temperature=temperature
            )

        results.append({
            "uid": uid,
            "dataset": sample.get("dataset"),
            "model": sample.get("model"),
            "human_severity": sample.get("human_severity"),
            "severity": severity,
            "raw_response": raw_response,
            "error": error,
        })

        if error:
            print(f"  [{i+1}/{n_total}] {uid}: ERROR - {error}")
        else:
            print(f"  [{i+1}/{n_total}] {uid}: severity={severity}")

        # save progress every sample
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(0.1 if is_local else 0.3)  # lighter rate-limit for API calls

    n_failed = sum(1 for r in results if r.get("severity") is None)
    print(f"  Done. {len(results)} total, {n_failed} failed to parse/error.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True,
                        choices=ALL_JUDGES + ["all"])
    parser.add_argument("--condition", required=True,
                        choices=list(PROMPTS.keys()) + ["all"])
    parser.add_argument("--pool-path", default=CANDIDATE_POOL_PATH)
    args = parser.parse_args()

    pool = load_candidate_pool(args.pool_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    judges = ALL_JUDGES if args.judge == "all" else [args.judge]
    conditions = list(PROMPTS.keys()) if args.condition == "all" else [args.condition]

    for judge_name in judges:
        for condition in conditions:
            output_path = os.path.join(RESULTS_DIR, f"{judge_name}_{condition}.json")
            run_judge_condition(judge_name, condition, pool, output_path)


if __name__ == "__main__":
    main()