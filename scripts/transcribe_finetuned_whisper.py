"""
scripts/transcribe_finetuned_whisper.py

Runs the winning fine-tuned checkpoint (whisper-medium + LoRA, r32,
lr5e4 - confirmed test WER 10.54%) across all 4 datasets, saving output
in the SAME canonical benchmark format find_canonical_file() expects
(model name, dataset, samples with ref/hyp/sample_WER/sample_index),
so this can be plugged into naive.py/context_v1.py/etc. as a genuine
5th ASR model alongside qwen/whisperx/parakeet/wav2vec2.

Runs on the FULL dataset (not just dev/test) for CommonVoice, EdAcc,
English Dialects - matching every other baseline benchmark file's
convention (get_indices_for_split() applies dev/test restriction
LATER, when this file is consumed by an ensemble script - it should
not be pre-restricted here). Shetland is always run in full, as usual.

KNOWN GAP: this does NOT produce word-level "segments"/confidence
data - only ref/hyp/sample_WER/sample_index. This means naive.py,
context_v1.py, context_v2.py etc. (which only need hyp text) will work
fine with this model added, but the *_confidence.py variants (which
need per-word confidence) will NOT work with this model included until
confidence extraction is added separately - flag this if you try to
add whisper_ft into a confidence-informed ensemble run.

Usage:
    python scripts/transcribe_finetuned_whisper.py --dataset commonvoice
    python scripts/transcribe_finetuned_whisper.py --dataset shetland
    python scripts/transcribe_finetuned_whisper.py --datasets commonvoice edacc english_dialects shetland
"""

import json
import os
import argparse
import time
from datetime import datetime

import torch
import soundfile as sf
from jiwer import wer as compute_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc, Shetland
from src.text_normalise import normalise

CHECKPOINT_DIR = "checkpoints/whisper_medium_lora_scots_r32_lr5e4_final/best"
BASE_MODEL = "openai/whisper-medium"
MODEL_KEY = "whisper_ft"   # the "model" name saved in output, and the key to use in ASR_MODELS lists downstream
OUTPUT_DIR = "writeup_results/benchmarks/main"
LANGUAGE = "english"
TASK = "transcribe"

DATASET_LOADERS = {
    "commonvoice": CommonVoiceScots,
    "english_dialects": EnglishDialectsScots,
    "edacc": EdAcc,
    "shetland": Shetland,
}


def load_model():
    print(f"Loading base model: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    print(f"Loading LoRA adapter from: {CHECKPOINT_DIR}")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    return model, processor, device


def transcribe_one(model, processor, device, audio, sr):
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)

    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=LANGUAGE,
            task=TASK,
            # Whisper's decoder has an absolute architectural ceiling of 448
            # positions (not configurable higher). The smoke test showed
            # the fine-tuning script's own default (225, generation_max_length)
            # truncating real, longer CommonVoice utterances mid-sentence -
            # inflating WER on samples that were otherwise transcribed
            # correctly. 440 leaves a small safety margin under the true
            # 448 ceiling for the initial forced task/language tokens.
            max_new_tokens=440,
        )

    text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_dataset(model, processor, device, dataset_key, max_samples=None):
    dataset = DATASET_LOADERS[dataset_key]()
    print(f"\n-- {dataset_key} --")

    samples_out = []
    refs_all, hyps_all = [], []

    start_time = time.time()
    for i, sample in enumerate(dataset.load()):
        if max_samples and i >= max_samples:
            break

        ref = sample.label
        audio = sample.audio
        sr = sample.sample_rate

        try:
            hyp = transcribe_one(model, processor, device, audio, sr)
        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            samples_out.append({"ref": ref, "hyp": None, "sample_WER": None,
                                "sample_index": i, "error": True, "error_reason": str(e)})
            continue

        sample_wer_val = compute_wer(normalise(ref), normalise(hyp)) if ref and hyp else None

        samples_out.append({
            "ref": ref,
            "hyp": hyp,
            "sample_WER": sample_wer_val,
            "sample_index": i,
        })

        if ref and hyp:
            refs_all.append(normalise(ref))
            hyps_all.append(normalise(hyp))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1} samples done ({elapsed:.0f}s elapsed)")

    corpus_wer = compute_wer(refs_all, hyps_all) if refs_all else None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{MODEL_KEY}_{dataset_key}_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    output = {
        "model": MODEL_KEY,
        "dataset": dataset_key,
        "checkpoint": CHECKPOINT_DIR,
        "base_model": BASE_MODEL,
        "corpus_wer": corpus_wer,
        "num_samples": len(samples_out),
        "samples": samples_out,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    print(f"  Corpus WER: {wer_str}  (N={len(refs_all)})")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset - smoke test only")
    args = parser.parse_args()

    if args.datasets:
        targets = args.datasets
    elif args.dataset:
        targets = [args.dataset]
    else:
        targets = list(DATASET_LOADERS.keys())

    model, processor, device = load_model()

    for dataset_key in targets:
        transcribe_dataset(model, processor, device, dataset_key, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
