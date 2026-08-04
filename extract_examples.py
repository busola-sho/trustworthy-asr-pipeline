import argparse
from src.datasets import CommonVoiceScots, EnglishDialectsScots, EdAcc, Shetland

DATASETS = {
    "commonvoice": CommonVoiceScots,
    "english_dialects": EnglishDialectsScots,
    "edacc": EdAcc,
    "shetland": Shetland,
}


def extract(dataset_key):
    dataset = DATASETS[dataset_key]()
    entries = []

    for sample in dataset.load():
        text = sample.label.strip()
        duration = len(sample.audio) / sample.sample_rate
        n_words = len(text.split())
        entries.append((duration, n_words, text))

    entries.sort(key=lambda x: x[0])  # sort by duration

    shortest = entries[0]
    longest = entries[-1]

    print(f"\n{'='*70}")
    print(f"  {dataset_key.upper()}")
    print(f"{'='*70}")
    print(f"\nSHORTEST (duration={shortest[0]:.2f}s, words={shortest[1]}):")
    print(f'  "{shortest[2]}"')
    print(f"\nLONGEST (duration={longest[0]:.2f}s, words={longest[1]}):")
    print(f'  "{longest[2]}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    for dataset_key in args.datasets:
        extract(dataset_key)


if __name__ == "__main__":
    main()
