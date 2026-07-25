"""
scripts/finetuning/finetune_wav2vec2.py

Fine-tunes a pretrained wav2vec2 checkpoint (CTC) on the train manifest,
selects the best checkpoint by validation WER, and ONLY evaluates on the
held-out test manifest if --evaluate_test is explicitly passed (same
reasoning as finetune_whisper_lora.py - see that script's docstring).

IMPORTANT CHANGE FROM THE FIRST VERSION: this now uses the base
checkpoint's OWN pretrained processor/vocabulary/CTC head by default,
rather than building a brand new character vocabulary and reinitialising
the output layer via ignore_mismatched_sizes. With only ~5-6 hours of
training audio, discarding the pretrained English CTC head throws away
most of the value of starting from a pretrained checkpoint - you'd
effectively be training a CTC head from scratch on a small dataset.
Only fall back to a custom vocabulary (--no-use_pretrained_vocab, see
build_wav2vec2_vocab.py) if your transcripts genuinely contain
characters the pretrained tokenizer can't represent.

Default base checkpoint is facebook/wav2vec2-large-960h-lv60-self - a
strong English-only starting point (LibriSpeech + LibriVox
self-training), appropriate for accent adaptation.

Data comes from prepare_manifest.py's output (data/finetune/manifests/
{train,val,test}.jsonl) - run that first.

Usage:
    # during hyperparameter search - val WER only, test never touched:
    python scripts/finetuning/finetune_wav2vec2.py \\
        --output_dir checkpoints/wav2vec2_scots \\
        --epochs 30 --lr 3e-5

    # final run once a config is chosen:
    python scripts/finetuning/finetune_wav2vec2.py \\
        --output_dir checkpoints/wav2vec2_scots_final \\
        --epochs 30 --lr 3e-5 --evaluate_test
"""

import argparse
import json
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from typing import Dict, List, Union

import torch
import soundfile as sf
from datasets import Dataset
from jiwer import wer
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from src.text_normalise import normalise

MANIFEST_DIR = "data/finetune/manifests"
VOCAB_PATH = "data/finetune/wav2vec2_vocab.json"   # only used if --no-use_pretrained_vocab
BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"
SEED = 42


def load_manifest_as_dataset(split: str) -> Dataset:
    path = f"{MANIFEST_DIR}/{split}.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return Dataset.from_list(entries)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def make_prepare_fn(processor: Wav2Vec2Processor):
    def prepare_example(example):
        # prepare_manifest.py already wrote 16kHz mono wavs - read
        # directly, bypassing datasets' Audio feature (requires
        # torchcodec as of recent `datasets` versions, avoided here)
        audio_array, sr = sf.read(example["audio_path"])
        example["input_values"] = processor(
            audio_array, sampling_rate=sr
        ).input_values[0]
        example["labels"] = processor(text=example["text"]).input_ids
        return example
    return prepare_example


def make_compute_metrics(processor: Wav2Vec2Processor):
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = pred_logits.argmax(axis=-1)

        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        error_rate = wer(
            [normalise(x) for x in label_str],
            [normalise(x) for x in pred_str],
        )
        return {"wer": error_rate}
    return compute_metrics


def get_package_versions() -> dict:
    packages = ["torch", "transformers", "datasets", "jiwer"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Full-model fine-tuning needs a much lower LR than LoRA - "
                             "3e-4 was too aggressive for the whole transformer stack; "
                             "start around 1e-5 to 5e-5 and compare via validation WER")
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False,
                        help="Off by default: the frozen-feature-encoder + gradient_checkpointing "
                             "interaction proved fragile in testing (silent zero-gradient training, "
                             "confirmed via smoke test: train_loss stuck at 0.0, eval_loss nan) even "
                             "with the standard forward-hook workaround in place. wav2vec2-large on a "
                             "full Ampere GPU with these batch sizes likely doesn't need the memory "
                             "savings anyway - only re-enable if you hit real OOM errors, and verify "
                             "with a smoke test (train_loss should be a real non-zero number, not "
                             "exactly 0.0) before trusting a real run.")
    parser.add_argument("--use_pretrained_vocab", action=argparse.BooleanOptionalAction, default=True,
                        help="Reuse the base checkpoint's own processor/vocab/CTC head "
                             "(recommended default for this data volume). Pass "
                             "--no-use_pretrained_vocab only if your transcripts contain "
                             "characters the pretrained tokenizer can't represent - "
                             "requires build_wav2vec2_vocab.py to have been run first, "
                             "and will reinitialise the CTC output head from scratch.")
    parser.add_argument("--freeze_feature_encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze the CNN feature encoder (recommended default for a "
                             "small fine-tuning set). Use --no-freeze_feature_encoder to "
                             "also adapt the feature encoder.")
    parser.add_argument("--evaluate_test", action="store_true",
                        help="Only set this on your FINAL run, once hyperparameters are "
                             "fixed via validation WER.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit train set to this many samples - smoke test only")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit val set to this many samples - smoke test only")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    if args.use_pretrained_vocab:
        print(f"Loading PRETRAINED processor/vocab/CTC head from {args.base_model} "
              f"(recommended for this data volume - keeps the existing English CTC head)")
        processor = Wav2Vec2Processor.from_pretrained(args.base_model)
        model = Wav2Vec2ForCTC.from_pretrained(
            args.base_model,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,   # prevents NaN loss when a sample's audio is too
                                      # short relative to its transcript for CTC alignment
                                      # to be possible - such samples contribute zero loss
                                      # instead of poisoning the whole batch's average
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    else:
        print(f"Loading CUSTOM vocab from {VOCAB_PATH} (run build_wav2vec2_vocab.py first) "
              f"- this REINITIALISES the CTC output head from scratch, only use if the "
              f"pretrained vocab genuinely can't represent your target text")
        tokenizer = Wav2Vec2CTCTokenizer(
            VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained(
            args.base_model,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=True,
            pad_token_id=tokenizer.pad_token_id,
            vocab_size=len(tokenizer),
            ignore_mismatched_sizes=True,
        )

    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        # Kept as a safety net if you re-enable --gradient_checkpointing
        # later (defaults OFF now - see that flag's help text for why).
        # This hook alone did NOT fully resolve the issue in testing, so
        # don't trust gradient_checkpointing=True without re-verifying
        # via smoke test (train_loss should be non-zero, eval_loss should
        # not be nan) even with this hook in place.
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.wav2vec2.feature_extractor.register_forward_hook(_make_inputs_require_grad)

    print("Loading manifests...")
    train_ds = load_manifest_as_dataset("train")
    val_ds = load_manifest_as_dataset("val")
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples:
        val_ds = val_ds.select(range(min(args.max_val_samples, len(val_ds))))
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    prepare_fn = make_prepare_fn(processor)
    train_ds = train_ds.map(prepare_fn, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(prepare_fn, remove_columns=val_ds.column_names, num_proc=1)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = make_compute_metrics(processor)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
        report_to=[],
        group_by_length=True,
        warmup_ratio=args.warmup_ratio,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    start_time = time.time()
    print("Starting training...")
    trainer.train()
    training_duration_sec = time.time() - start_time

    print(f"Saving best checkpoint (by val WER) to {args.output_dir}/best")
    trainer.save_model(f"{args.output_dir}/best")
    processor.save_pretrained(f"{args.output_dir}/best")

    best_val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    print(f"Best validation metrics: {best_val_metrics}")

    run_metadata = {
        "base_model": args.base_model,
        "use_pretrained_vocab": args.use_pretrained_vocab,
        "freeze_feature_encoder": args.freeze_feature_encoder,
        "gradient_checkpointing": args.gradient_checkpointing,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "seed": SEED,
        "package_versions": get_package_versions(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
        "training_duration_sec": round(training_duration_sec, 1),
        "best_val_metrics": best_val_metrics,
        "command_line_args": vars(args),
    }

    if args.evaluate_test:
        print("\n--evaluate_test set: evaluating best checkpoint on held-out TEST manifest...")
        test_ds = load_manifest_as_dataset("test")
        test_ds = test_ds.map(prepare_fn, remove_columns=test_ds.column_names, num_proc=1)
        test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        print(f"Test set results: {test_metrics}")
        run_metadata["test_metrics"] = test_metrics
    else:
        print("\n--evaluate_test NOT set - test manifest untouched, as intended during "
              "hyperparameter search.")

    with open(f"{args.output_dir}/run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved run metadata -> {args.output_dir}/run_metadata.json")


if __name__ == "__main__":
    main()