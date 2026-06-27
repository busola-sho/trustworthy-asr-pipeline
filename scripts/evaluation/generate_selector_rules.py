"""generate_selector_rules.py — CLI wrapper around src.rules"""
import argparse
from src.rules import build_rules_text, load_profiles, discover_models_and_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap",     type=float, default=1.0)
    parser.add_argument("--list",    action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    profiles = load_profiles()
    models, datasets, _ = discover_models_and_datasets(profiles)

    if args.list:
        print(f"Models:   {models}")
        print(f"Datasets: {datasets}")
        return

    rules = build_rules_text(min_gap_pp=args.gap, save=not args.no_save)
    print(rules)

if __name__ == "__main__":
    main()