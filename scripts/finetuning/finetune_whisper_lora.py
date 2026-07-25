"""
scripts/finetuning/finetune_whisper_lora.py

Fine-tunes openai/whisper-small with LoRA on the train manifest, selects
the best checkpoint by validation WER, evaluates on the held-out test
manifest at the end.

Data comes from prepare_manifest.py's output (data/finetune/manifests/
{train,val,test}.jsonl) - run that first.

Usage:
    python scripts/finetuning/finetune_whisper_lora.py \\
        --output_dir checkpoints/whisper_small_lora_scots \\
        --lora_r 32 --lora_alpha 64 --epochs 10 --lr 1e-3

    # try a different LoRA config as a second run for comparison:
    python scripts/finetuning/finetune_whisper_lora.py \\
        --output_dir checkpoints/whisper_small_lora_scots_r16 \\
        --lora_r 16 --lora_alpha 32 --epochs 10 --lr 1e-3
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import soundfile as sf
from datasets import Dataset, Audio
from jiwer import wer
from peft import LoraConfig, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MANIFEST_DIR = "data/finetune/manifests"
BASE_MODEL = "openai/whisper-small"
LANGUAGE = "english"
TASK = "transcribe"


def load_manifest_as_dataset(split: str) -> Dataset:
    path = f"{MANIFEST_DIR}/{split}.jsonl"
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    ds = Dataset.from_list(entries)
    ds = ds.rename_column("audio_path", "audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos was appended by the tokenizer during pad, strip it here since
        # the model prepends it automatically
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def make_prepare_fn(processor: WhisperProcessor):
    def prepare_example(example):
        audio = example["audio"]
        example["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
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

        error_rate = wer(label_str, pred_str)
        return {"wer": error_rate}
    return compute_metrics


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
    parser.add_argument("--eval_steps", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading processor/model: {args.base_model}")
    processor = WhisperProcessor.from_pretrained(args.base_model, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading manifests...")
    train_ds = load_manifest_as_dataset("train")
    val_ds = load_manifest_as_dataset("val")
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    prepare_fn = make_prepare_fn(processor)
    train_ds = train_ds.map(prepare_fn, remove_columns=train_ds.column_names, num_proc=1)
    val_ds = val_ds.map(prepare_fn, remove_columns=val_ds.column_names, num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best checkpoint (by val WER) to {args.output_dir}/best")
    trainer.save_model(f"{args.output_dir}/best")
    processor.save_pretrained(f"{args.output_dir}/best")

    print("\nEvaluating best checkpoint on held-out TEST manifest...")
    test_ds = load_manifest_as_dataset("test")
    test_ds = test_ds.map(prepare_fn, remove_columns=test_ds.column_names, num_proc=1)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(f"Test set results: {test_metrics}")

    with open(f"{args.output_dir}/test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()