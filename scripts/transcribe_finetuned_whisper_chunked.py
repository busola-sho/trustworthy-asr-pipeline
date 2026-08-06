"""
scripts/transcribe_finetuned_whisper_chunked.py

FIX over transcribe_finetuned_whisper.py: raising max_new_tokens (225
-> 440) had ZERO effect on the observed truncation (byte-identical
output, same 93-word stopping point every time) - proving the model
was never actually being cut off by a token ceiling. It was genuinely
CHOOSING to stop early (predicting EOS), most likely because every
training example was <=28s (per prepare_manifest.py's own chunking
logic) - the model never saw a longer continuous example during
fine-tuning, so it has no learned behavior for transcribing past that
length in one pass.

This version chunks audio into ~28s windows AT INFERENCE TIME (same
MAX_CHUNK_SEC as training), transcribes each chunk separately, and
concatenates the results - matching training conditions, rather than
feeding the model out-of-distribution long-form audio in one shot.

Simple time-based chunking (no VAD/silence detection) - a chunk
boundary may occasionally land mid-word, a minor tradeoff worth
testing first before adding more sophisticated boundary detection.

Usage:
    python scripts/transcribe_finetuned_whisper_chunked.py --dataset commonvoice --max-samples 5
"""

import json
import os
import argparse
import time
from datetime import datetime

import torch
import numpy as np
from jiwer import wer as compute_wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc, Shetland

import subprocess
import pandas as pd

# The system module's ffmpeg (loaded via `module load ffmpeg/6.0`) is
# dynamically linked against libva.so.1 (VA-API hardware VIDEO
# acceleration) - confirmed missing on this GPU compute node's OS
# image (error while loading shared libraries: libva.so.1: cannot open
# shared object file). We only need audio decoding, so this dependency
# is irrelevant to begin with - imageio_ffmpeg bundles a fully STATIC
# ffmpeg binary with zero system library dependencies, sidestepping
# the problem entirely rather than trying to get libva installed.
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    raise ImportError(
        "imageio_ffmpeg is required for audio decoding on this node "
        "(the system ffmpeg module is missing libva.so.1). Install with:\n"
        "  pip install imageio-ffmpeg --break-system-packages"
    )


def load_audio_via_ffmpeg(path, target_sr=16000):
    """Decodes any audio file (including .m4a/AAC) via a direct ffmpeg
    subprocess call, bypassing librosa/audioread entirely. Used
    specifically for Shetland, whose .m4a files fail to decode through
    soundfile/audioread in this environment (confirmed: NoBackendError,
    even with the system ffmpeg binary present and a fresh audioread
    install - the Python-level AAC decoding chain is broken here, not
    the ffmpeg binary itself). ffmpeg's own decoder handles AAC/.m4a
    natively and reliably, so this sidesteps the problem completely.

    Outputs raw 32-bit float PCM, mono, at target_sr, piped directly
    from ffmpeg's stdout - no temp files needed.
    """
    cmd = [
        FFMPEG_BIN, "-i", path,
        "-f", "f32le", "-ar", str(target_sr), "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio_array = np.frombuffer(result.stdout, dtype=np.float32)
    return audio_array, target_sr


def load_shetland_via_ffmpeg(path="data/shetland"):
    """Self-contained Shetland loader mirroring src/datasets.py's
    Shetland class exactly (same excel columns, same skip logic), but
    using load_audio_via_ffmpeg() instead of librosa.load() for the
    audio decoding step. Yields (ref, audio_array, sample_rate) tuples,
    matching what transcribe_dataset() below expects from a Sample."""
    audio_dir = os.path.join(path, "audios")
    excel_path = os.path.join(path, "shetland.xlsx")

    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["transcript"])

    for _, row in df.iterrows():
        audio_file = str(row["audio_file"]).strip()
        transcript = str(row["transcript"]).strip()

        if not transcript or transcript == "nan":
            continue

        audio_path = os.path.join(audio_dir, audio_file)
        if not os.path.exists(audio_path):
            continue

        audio_array, sample_rate = load_audio_via_ffmpeg(audio_path)
        yield transcript, audio_array, sample_rate

def load_audio_bytes_via_ffmpeg(audio_bytes, target_sr=16000):
    """Same idea as load_audio_via_ffmpeg(), but for in-memory raw file
    bytes (piped to ffmpeg via stdin) rather than a filesystem path -
    needed for HF datasets' Audio(decode=False) results, which may
    provide 'bytes' with no usable 'path' depending on how the dataset
    was cached."""
    cmd = [
        FFMPEG_BIN, "-i", "pipe:0",
        "-f", "f32le", "-ar", str(target_sr), "-ac", "1",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=True)
    audio_array = np.frombuffer(result.stdout, dtype=np.float32)
    return audio_array, target_sr


def _decode_hf_audio_entry(audio_entry):
    """HF's Audio(decode=False) returns a dict with 'path' and/or
    'bytes' - prefer path (avoids buffering the whole file into
    memory) when a real, existing local file is available, otherwise
    fall back to the raw bytes."""
    path = audio_entry.get("path")
    if path and os.path.exists(path):
        return load_audio_via_ffmpeg(path)
    raw_bytes = audio_entry.get("bytes")
    if raw_bytes:
        return load_audio_bytes_via_ffmpeg(raw_bytes)
    raise ValueError(f"HF audio entry has neither a usable path nor bytes: {audio_entry.keys()}")


def load_english_dialects_via_ffmpeg():
    """Self-contained English Dialects loader mirroring
    src/datasets.py's EnglishDialectsScots class exactly, but disabling
    HF datasets' automatic Audio decoding (Audio(decode=False)) - which
    would otherwise require torchcodec, broken in this environment
    (confirmed: OSError loading libtorchcodec_core4.so, a build/version
    mismatch, not fixable by just reinstalling the package) - and
    decoding manually via ffmpeg instead. Yields (ref, audio_array,
    sample_rate) tuples."""
    from datasets import load_dataset, Audio
    from itertools import chain

    dataset_f = load_dataset("ylacombe/english_dialects", "scottish_female", split="train")
    dataset_m = load_dataset("ylacombe/english_dialects", "scottish_male", split="train")
    dataset_f = dataset_f.cast_column("audio", Audio(decode=False))
    dataset_m = dataset_m.cast_column("audio", Audio(decode=False))
    combined = chain(dataset_f, dataset_m)

    for row in combined:
        audio_array, sample_rate = _decode_hf_audio_entry(row["audio"])
        yield row["text"], audio_array, sample_rate


def load_edacc_via_ffmpeg():
    """Same treatment as load_english_dialects_via_ffmpeg(), mirroring
    src/datasets.py's EdAcc class (same accent filter, same IGNORE-text
    skip). Yields (ref, audio_array, sample_rate) tuples."""
    from datasets import load_dataset, Audio

    dataset = load_dataset("edinburghcstr/edacc", split="test")
    dataset = dataset.cast_column("audio", Audio(decode=False))

    for row in dataset:
        if row.get("accent") != "Scottish English":
            continue
        if row.get("text", "").startswith("IGNORE"):
            continue
        audio_array, sample_rate = _decode_hf_audio_entry(row["audio"])
        yield row["text"], audio_array, sample_rate


from src.text_normalise import normalise

CHECKPOINT_DIR = "checkpoints/whisper_medium_lora_scots_r32_lr5e4_final/best"
BASE_MODEL = "openai/whisper-medium"
MODEL_KEY = "whisper_ft_chunked"
OUTPUT_DIR = "writeup_results/benchmarks/main"
LANGUAGE = "english"
TASK = "transcribe"
MAX_CHUNK_SEC = 28   # same threshold as prepare_manifest.py's training-time chunking
TARGET_SR = 16000

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


def chunk_audio(audio, sr, max_chunk_sec=MAX_CHUNK_SEC):
    """Simple time-based chunking, no VAD - splits into fixed-length
    windows. A chunk boundary may occasionally land mid-word."""
    chunk_samples = int(max_chunk_sec * sr)
    if len(audio) <= chunk_samples:
        return [audio]
    chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


def transcribe_chunk(model, processor, device, audio_chunk, sr):
    if sr != TARGET_SR:
        import librosa
        audio_chunk = librosa.resample(audio_chunk.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)

    inputs = processor.feature_extractor(audio_chunk, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language=LANGUAGE,
            task=TASK,
            max_new_tokens=440,
        )

    text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_one(model, processor, device, audio, sr):
    chunks = chunk_audio(audio, sr)
    texts = []
    for chunk in chunks:
        text = transcribe_chunk(model, processor, device, chunk, sr)
        if text:
            texts.append(text)
    return " ".join(texts), len(chunks)


def transcribe_dataset(model, processor, device, dataset_key, max_samples=None, rerun=False):
    dataset = DATASET_LOADERS[dataset_key]()
    print(f"\n-- {dataset_key} --")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # fixed filename (no timestamp) so a resumed run can find its own
    # progress - every other script tonight follows this same pattern
    filename = f"{MODEL_KEY}_{dataset_key}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path) and not rerun:
        with open(output_path) as f:
            existing = json.load(f)
        samples_out = existing.get("samples", [])
        start_from = len(samples_out)
        print(f"  Resuming from sample {start_from}")
    else:
        samples_out = []
        start_from = 0

    def save_progress():
        refs_so_far = [normalise(s["ref"]) for s in samples_out if s.get("ref") and s.get("hyp")]
        hyps_so_far = [normalise(s["hyp"]) for s in samples_out if s.get("ref") and s.get("hyp")]
        corpus_wer_so_far = compute_wer(refs_so_far, hyps_so_far) if refs_so_far else None
        output = {
            "model": MODEL_KEY,
            "dataset": dataset_key,
            "checkpoint": CHECKPOINT_DIR,
            "base_model": BASE_MODEL,
            "chunking": f"{MAX_CHUNK_SEC}s time-based chunks, no VAD",
            "corpus_wer": corpus_wer_so_far,
            "num_samples": len(samples_out),
            "samples": samples_out,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    start_time = time.time()

    # datasets with a broken decode path in .venv_finetune get routed
    # through the ffmpeg-subprocess bypass loaders defined above -
    # CommonVoice's plain librosa.load() already works fine here
    # (confirmed via earlier successful smoke tests), so it uses the
    # normal shared loader unchanged.
    BYPASS_LOADERS = {
        "shetland": load_shetland_via_ffmpeg,
        "english_dialects": load_english_dialects_via_ffmpeg,
        "edacc": load_edacc_via_ffmpeg,
    }

    if dataset_key in BYPASS_LOADERS:
        sample_iter = ((ref, audio, sr) for ref, audio, sr in BYPASS_LOADERS[dataset_key]())
    else:
        sample_iter = ((s.label, s.audio, s.sample_rate) for s in dataset.load())

    for i, (ref, audio, sr) in enumerate(sample_iter):
        if i < start_from:
            continue
        if max_samples and i >= max_samples:
            break

        try:
            hyp, n_chunks = transcribe_one(model, processor, device, audio, sr)
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
            "n_chunks": n_chunks,
        })

        if (i + 1) % 10 == 0:
            save_progress()
            elapsed = time.time() - start_time
            print(f"  {i+1} samples done ({elapsed:.0f}s elapsed)")

    save_progress()

    refs_all = [normalise(s["ref"]) for s in samples_out if s.get("ref") and s.get("hyp")]
    corpus_wer = compute_wer(refs_all, [normalise(s["hyp"]) for s in samples_out if s.get("ref") and s.get("hyp")]) if refs_all else None
    wer_str = f"{corpus_wer*100:.2f}%" if corpus_wer is not None else "-"
    print(f"  Corpus WER: {wer_str}  (N={len(refs_all)})")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rerun", action="store_true",
                        help="Start fresh instead of resuming from an existing output file")
    args = parser.parse_args()

    if args.datasets:
        targets = args.datasets
    elif args.dataset:
        targets = [args.dataset]
    else:
        targets = list(DATASET_LOADERS.keys())

    model, processor, device = load_model()

    for dataset_key in targets:
        transcribe_dataset(model, processor, device, dataset_key, max_samples=args.max_samples, rerun=args.rerun)


if __name__ == "__main__":
    main()
