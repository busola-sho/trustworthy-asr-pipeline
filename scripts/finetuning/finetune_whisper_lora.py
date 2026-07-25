"""
scripts/finetuning/finetune_whisper_lora.py

Fine-tunes openai/whisper-small with LoRA on the train manifest, selects
the best checkpoint by validation WER, and ONLY evaluates on the
held-out test manifest if --evaluate_test is explicitly passed (see
below - this prevents accidentally using test performance to choose
hyperparameters during a sweep).

Data comes from prepare_manifest.py's output (data/finetune/manifests/
{train,val,test}.jsonl) - run that first.

WORKFLOW (per the reviewed feedback):
  1. Train on train.jsonl.
  2. Select checkpoints/hyperparameters using val.jsonl ONLY - run
     without --evaluate_test while you're still comparing LoRA configs
     or learning rates.
  3. Once a configuration is fixed, run ONE final job with
     --evaluate_test to get the test-set number. Do this once, not
     per-sweep-iteration.
  4. Shetland stays untouched - handled elsewhere, not by this script.

Usage:
    # during hyperparameter search - val WER only, test never touched:
    python scripts/finetuning/finetune_whisper_lora.py \\
        --output_dir checkpoints/whisper_small_lora_scots_r32 \\
        --lora_r 32 --lora_alpha 64 --epochs 10 --lr 1e-3

    # final run once a config is chosen - evaluates test ONCE:
    python scripts/finetuning/finetune_whisper_lora.py \\
        --output_dir checkpoints/whisper_small_lora_scots_final \\
        --lora_r 32 --lora_alpha 64 --epochs 10 --lr 1e-3 --evaluate_test
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
from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from src.text_normalise import normalise

MANIFEST_DIR = "data/finetune/manifests"
BASE_MODEL = "openai/whisper-small"
LANGUAGE = "english"
TASK = "transcribe"
SEED = 42


def load_manifest_as_dataset(split: str) -> Dataset:
    path = f"{MANIFEST_DIR}/{split}.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return Dataset.from_list(entries)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_prepare_fn(processor: WhisperProcessor):
    def prepare_example(example):
        # prepare_manifest.py already wrote 16kHz mono wavs, so no
        # resampling needed here - read directly, bypassing datasets'
        # Audio feature (which now requires torchcodec as of recent
        # `datasets` versions - avoided here to keep the fine-tuning env lean)
        audio_array, sr = sf.read(example["audio_path"])
        example["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example
    return prepare_example


def make_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # normalise() matches the rest of the dissertation pipeline, so this
        # validation WER is directly comparable to your other reported WERs -
        # NOT the same number you'd get from raw string comparison.
        error_rate = wer(
            [normalise(x) for x in label_str],
            [normalise(x) for x in pred_str],
        )
        return {"wer": error_rate}
    return compute_metrics


def get_package_versions() -> dict:
    packages = ["torch", "transformers", "peft", "datasets", "jiwer"]
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
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--generation_max_length", type=int, default=225,
                        help="Check your split_summary.json / manifest target-token-length "
                             "distribution before trusting this default for long-tailed data")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--evaluate_test", action="store_true",
                        help="Only set this on your FINAL run, once hyperparameters are "
                             "fixed via validation WER - evaluating test repeatedly during "
                             "a sweep defeats the point of holding it out.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit train set to this many samples - smoke test only")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit val set to this many samples - smoke test only")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    print(f"Loading processor/model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False   # required alongside gradient checkpointing

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()   # required for LoRA + gradient_checkpointing to work
                                          # correctly - without this, gradients may not properly
                                          # flow into the (frozen-base) LoRA adapters, and the
                                          # "None of the inputs have requires_grad=True" warning
                                          # can mean training is silently a no-op
    model.print_trainable_parameters()

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

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Seq2SeqTrainer(
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
    trainer.save_model(f"{args.output_dir}/best")   # saves the PEFT adapter, not a merged model
    processor.save_pretrained(f"{args.output_dir}/best")

    best_val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="val")
    print(f"Best validation metrics: {best_val_metrics}")

    run_metadata = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "learning_rate": args.lr,
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
              "hyperparameter search. Re-run with --evaluate_test once you've picked a "
              "final configuration.")

    with open(f"{args.output_dir}/run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Saved run metadata -> {args.output_dir}/run_metadata.json")


if __name__ == "__main__":
    main()