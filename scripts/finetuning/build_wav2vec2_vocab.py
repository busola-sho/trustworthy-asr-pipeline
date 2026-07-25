"""
scripts/finetuning/build_wav2vec2_vocab.py

Builds a character-level vocabulary for the Wav2Vec2CTCTokenizer, from
the TRAIN manifest's transcripts only (never val/test - the tokenizer's
vocab shouldn't be informed by held-out data). Run this once before
finetune_wav2vec2.py.

Usage:
    python scripts/finetuning/build_wav2vec2_vocab.py
"""

import json

MANIFEST_PATH = "data/finetune/manifests/train.jsonl"
VOCAB_OUTPUT_PATH = "data/finetune/wav2vec2_vocab.json"


def main():
    chars = set()
    with open(MANIFEST_PATH) as f:
        for line in f:
            entry = json.loads(line)
            chars.update(entry["text"])

    # word_delimiter_token="|" (space) is handled specially, so remove
    # literal spaces from the char set and represent them with "|" instead
    chars.discard(" ")

    vocab_list = sorted(chars)
    vocab_dict = {v: i for i, v in enumerate(vocab_list)}

    # add special tokens
    vocab_dict["|"] = len(vocab_dict)      # word delimiter (represents space)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(VOCAB_OUTPUT_PATH, "w") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"Built vocab with {len(vocab_dict)} tokens from {MANIFEST_PATH}")
    print(f"Saved: {VOCAB_OUTPUT_PATH}")


if __name__ == "__main__":
    main()

