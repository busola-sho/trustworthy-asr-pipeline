"""
scripts/finetuning/finetune_wav2vec2.py

Fine-tunes a pretrained wav2vec2 checkpoint (CTC) on the train manifest,
selects the best checkpoint by validation WER, evaluates on the held-out
test manifest at the end.

Default base checkpoint is facebook/wav2vec2-large-960h-lv60-self - a
strong English-only starting point (LibriSpeech + LibriVox self-training),
appropriate for accent adaptation rather than needing wav2vec2's
multilingual XLSR variant. Override with --base_model if you'd rather
start from XLSR-53 or a different checkpoint.

Data comes from prepare_manifest.py's output (data/finetune/manifests/
{train,val,test}.jsonl) - run that first, and run build_wav2vec2_vocab.py
once before this script (needed to build the character-level tokenizer
vocab from your training transcripts).

Usage:
    python scripts/finetuning/build_wav2vec2_vocab.py
    python scripts/finetuning/finetune_wav2vec2.py \\
        --output_dir checkpoints/wav2vec2_scots \\
        --epochs 30 --lr 3e-4
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from datasets import Dataset, Audio
from jiwer import wer
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

MANIFEST_DIR = "data/finetune/manifests"
VOCAB_PATH = "data/finetune/wav2vec2_vocab.json"
BASE_MODEL = "facebook/wav2vec2-large-960h-lv60-self"


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
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def make_prepare_fn(processor: Wav2Vec2Processor):
    def prepare_example(example):
        audio = example["audio"]
        example["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        with processor.as_target_processor():
            example["labels"] = processor(example["text"]).input_ids
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

        error_rate = wer(label_str, pred_str)
        return {"wer": error_rate}
    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--freeze_feature_encoder", action="store_true", default=True,
                        help="Freeze the CNN feature encoder (recommended default for "
                             "small fine-tuning sets - only adapt the transformer + CTC head)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit train set to this many samples - for a quick smoke "
                             "test before submitting a real job, not for actual training runs")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit val set to this many samples - smoke-test only")
    args = parser.parse_args()

    print(f"Loading tokenizer from {VOCAB_PATH} (run build_wav2vec2_vocab.py first if missing)")
    tokenizer = Wav2Vec2CTCTokenizer(
        VOCAB_PATH, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print(f"Loading base model: {args.base_model}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

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

    training_args = TrainingArguments(
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
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
        group_by_length=True,
        warmup_steps=500,
    )

    trainer = Trainer(
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