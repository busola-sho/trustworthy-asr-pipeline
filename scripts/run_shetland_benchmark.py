"""
run_shetland_benchmark.py

Runs all 4 ASR models + Qwen P2 MAR judge on the Shetland self-collated dataset.
Reads from data/shetland/shetland.xlsx and data/shetland/audios/.

Resume-friendly — saves after every sample, resumes from existing file.

Output:
    benchmarks/shetland_{model}_{timestamp}.json

Usage:
    python scripts/run_shetland_benchmark.py --model qwen3asr
    python scripts/run_shetland_benchmark.py --model all
    python scripts/run_shetland_benchmark.py --model whisper --max_samples 10
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from jiwer import wer
from ollama import Client

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR       = "data/shetland"
AUDIO_DIR      = os.path.join(DATA_DIR, "audios")
EXCEL_PATH     = os.path.join(DATA_DIR, "shetland.xlsx")
BENCHMARKS_DIR = "benchmarks"
OLLAMA_HOST    = "http://localhost:11434"
QWEN_MODEL     = "qwen2.5:7b"

MODELS = {
    "whisper":  "openai/whisper-large-v3",
    "wav2vec2": "facebook/wav2vec2-large-960h-lv60-self",
    "parakeet": "nvidia/parakeet-ctc-1.1b",
    "qwen3asr": "Qwen/Qwen3-ASR-1.7B",
}

MAR_PROMPT = """You are evaluating ASR transcripts for a policing context. Given a reference and hypothesis transcript of the same audio, determine if the hypothesis contains meaning-altering errors that would cause a police officer to misunderstand what was said.

Ignore: capitalisation, punctuation, contractions, dialect normalisation (e.g. "didnae"→"didn't", "oot"→"out"), filler words.

Flag as meaning-altering if:
- Factual content changes
- Negation is added or removed
- A name, place, or number is wrong
- A dialect word is misrecognised as a different real word (e.g. "bairn"→"barn")
- Content is hallucinated over inaudible segments

Examples:
Reference: She said she wisnae near the pub on Saturday night.
Hypothesis: She said she was near the pub on Saturday night.
Reasoning: Negation "wisnae" dropped, reversing the speaker's alibi.
Answer: true

Reference: He works the back shift at the factory on Keppoch Road.
Hypothesis: He works the back shift at the factory on Keppoch Rd.
Reasoning: "Rd" is a standard abbreviation for Road; same location, no factual content lost.
Answer: false

Reply with only: true or false"""

# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── MAR judge ──────────────────────────────────────────────────────────────────

def run_mar_judge(client: Client, ref: str, hyp: str, sample_wer: float):
    if sample_wer == 0:
        return False
    try:
        response = client.chat(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": MAR_PROMPT},
                {"role": "user",   "content": f"Reference: {ref}\nHypothesis: {hyp}"},
            ],
            options={"temperature": 0},
        )
        result = response.message.content.strip().lower()
        if result.startswith("true"):
            return True
        elif result.startswith("false"):
            return False
        elif "true" in result and "false" not in result:
            return True
        elif "false" in result and "true" not in result:
            return False
        print(f"  WARNING: could not parse verdict: '{result[:80]}'")
        return None
    except Exception as e:
        print(f"  ERROR (MAR judge): {e}")
        return None

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_key: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC

    model_name = MODELS[model_key]
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"  Loading {model_key} on {device}...")

    if model_key == "whisper":
        model     = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float32).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return {"model": model, "processor": processor, "device": device, "type": "whisper"}

    elif model_key in ["wav2vec2", "parakeet"]:
        model     = AutoModelForCTC.from_pretrained(model_name, dtype=torch.float32).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
        return {"model": model, "processor": processor, "device": device, "type": "ctc"}

    elif model_key == "qwen3asr":
        from qwen_asr import Qwen3ASRModel
        qwen = Qwen3ASRModel.from_pretrained(model_name, torch_dtype=torch.float32)
        return {"model": qwen, "device": device, "type": "qwen"}

def transcribe(model_dict: dict, audio: np.ndarray) -> str:
    import torch

    model_type = model_dict["type"]
    device     = model_dict["device"]

    if model_type == "whisper":
        model     = model_dict["model"]
        processor = model_dict["processor"]
        chunk_length = 30 * 16000
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        full_text = ""
        for chunk in chunks:
            features = processor(
                chunk, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device).to(model.dtype)
            with torch.no_grad():
                tokens = model.generate(features, language="en")
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            full_text += " " + text
        return full_text.strip()

    elif model_type == "ctc":
        model     = model_dict["model"]
        processor = model_dict["processor"]
        chunk_length = 30 * 16000
        chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
        full_text = ""
        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(predicted_ids)[0]
            full_text += " " + text
        return full_text.strip()

    elif model_type == "qwen":
        results = model_dict["model"].transcribe((audio, 16000), language="English")
        return results[0].text

# ── Main runner ────────────────────────────────────────────────────────────────

def run_model(model_key: str, max_samples: int = None):
    print(f"\n{'='*60}")
    print(f"Model: {model_key} | Dataset: Shetland")
    print(f"{'='*60}")

    # load excel
    df = pd.read_excel(EXCEL_PATH, dtype={"clip_id": str, "audio_file": str})
    print(f"Loaded {len(df)} samples from {EXCEL_PATH}")

    if max_samples:
        df = df.head(max_samples)

    # resume: find existing file for this model
    existing_files = sorted([
        f for f in os.listdir(BENCHMARKS_DIR)
        if f.startswith(f"shetland_{model_key}_") and f.endswith(".json")
    ])

    if existing_files:
        output_path = os.path.join(BENCHMARKS_DIR, existing_files[-1])
        with open(output_path) as f:
            existing = json.load(f)
        results    = existing.get("samples", [])
        start_from = len(results)
        print(f"Resuming from sample {start_from} ({output_path})")
    else:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(BENCHMARKS_DIR, f"shetland_{model_key}_{timestamp}.json")
        results     = []
        start_from  = 0
        print(f"Starting fresh: {output_path}")

    # load model
    model_dict = load_model(model_key)
    print(f"  {model_key} loaded.")

    # connect to Ollama for MAR
    ollama_client = Client(host=OLLAMA_HOST)
    try:
        available     = [m.model for m in ollama_client.list().models]
        mar_available = any(QWEN_MODEL in m for m in available)
        if not mar_available:
            print(f"  WARNING: {QWEN_MODEL} not available — MAR will be skipped")
    except Exception:
        mar_available = False
        print("  WARNING: Ollama not reachable — MAR will be skipped")

    for idx, (i, row) in enumerate(df.iterrows()):
        if idx < start_from:
            continue

        clip_id    = row["clip_id"]
        ref        = str(row["transcript"]).strip()
        audio_file = str(row["audio_file"]).strip()
        audio_path = os.path.join(AUDIO_DIR, audio_file)

        if not os.path.exists(audio_path):
            print(f"  WARNING: missing audio {audio_path} — skipping")
            results.append({"clip_id": clip_id, "ref": ref, "hyp": None,
                            "sample_WER": None, "qwen_verdict_p2": None,
                            "error": "missing_audio"})
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"  ERROR loading {audio_path}: {e}")
            results.append({"clip_id": clip_id, "ref": ref, "hyp": None,
                            "sample_WER": None, "qwen_verdict_p2": None,
                            "error": str(e)})
            continue

        try:
            hyp = transcribe(model_dict, audio)
        except Exception as e:
            print(f"  ERROR transcribing {clip_id}: {e}")
            hyp = ""

        ref_norm   = normalise(ref)
        hyp_norm   = normalise(hyp)
        sample_wer_val = wer(ref_norm, hyp_norm) if ref_norm else 0.0

        mar_verdict = None
        if mar_available:
            mar_verdict = run_mar_judge(ollama_client, ref, hyp, sample_wer_val)
            time.sleep(0.1)

        results.append({
            "clip_id":         clip_id,
            "ref":             ref,
            "hyp":             hyp,
            "sample_WER":      sample_wer_val,
            "qwen_verdict_p2": mar_verdict,
        })

        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f,
                      indent=2, ensure_ascii=False)

        if len(results) % 10 == 0:
            print(f"  {len(results)}/{len(df)} done")

    # final metrics
    valid      = [r for r in results if r.get("sample_WER") is not None and not r.get("error")]
    corpus_wer_val = wer(
        [normalise(r["ref"]) for r in valid],
        [normalise(r["hyp"]) for r in valid]
    ) if valid else None
    mar = (sum(1 for r in valid if r.get("qwen_verdict_p2")) / len(valid)) if valid else None

    output = {
        "model":                   model_key,
        "model_name":              MODELS[model_key],
        "dataset":                 "shetland",
        "corpus_wer":              corpus_wer_val,
        "meaning_alteration_rate": mar,
        "num_samples":             len(valid),
        "samples":                 results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  Corpus WER: {corpus_wer_val*100:.2f}%" if corpus_wer_val is not None else "  Corpus WER: N/A")
    print(f"  MAR:        {mar*100:.2f}%"            if mar            is not None else "  MAR: N/A")
    print(f"  Saved to:   {output_path}")

    return output

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(BENCHMARKS_DIR, exist_ok=True)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    summary = []
    for model_key in models_to_run:
        result = run_model(model_key, args.max_samples)
        summary.append(result)

    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY — Shetland Dataset")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'WER':>8} {'MAR':>8} {'N':>6}")
        for r in summary:
            wer_str = f"{r['corpus_wer']*100:.2f}%" if r['corpus_wer'] is not None else "N/A"
            mar_str = f"{r['meaning_alteration_rate']*100:.2f}%" if r['meaning_alteration_rate'] is not None else "N/A"
            print(f"{r['model']:<12} {wer_str:>8} {mar_str:>8} {r['num_samples']:>6}")

if __name__ == "__main__":
    main()