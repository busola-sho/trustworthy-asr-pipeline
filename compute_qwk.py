"""
compute_qwk.py

Quadratic Weighted Kappa (QWK) between:
  1. Your own annotations vs. Yishun's annotations (genuine human-
     human inter-annotator agreement)
  2. Your own annotations vs. the judge's severity scores (human-AI
     agreement - what you already have)

Comparing these two numbers is the actual validation argument: if
judge-vs-you is comparable to you-vs-Yishun, that's strong evidence
the judge is about as reliable as a second human annotator would be.

Usage (as a library - adapt load_your_data() to your actual file
format, then call compute_qwk() on the aligned score lists):

    from compute_qwk import compute_qwk_report

    # each dict: {sample_id: severity_score}
    my_scores = {...}
    yishun_scores = {...}
    judge_scores = {...}

    compute_qwk_report(my_scores, yishun_scores, judge_scores)
"""

from sklearn.metrics import cohen_kappa_score


def align_scores(scores_a, scores_b):
    """Aligns two {id: score} dicts by shared keys only - reports how
    many IDs were dropped due to not appearing in both, since a
    silent mismatch here would corrupt the QWK calculation."""
    common_ids = set(scores_a.keys()) & set(scores_b.keys())
    only_a = set(scores_a.keys()) - common_ids
    only_b = set(scores_b.keys()) - common_ids

    if only_a or only_b:
        print(f"  WARNING: {len(only_a)} IDs only in first set, "
              f"{len(only_b)} IDs only in second set - excluded from QWK")

    aligned_ids = sorted(common_ids)
    a_vals = [scores_a[i] for i in aligned_ids]
    b_vals = [scores_b[i] for i in aligned_ids]
    return a_vals, b_vals, len(aligned_ids)


def compute_qwk(scores_a, scores_b, label_a="A", label_b="B"):
    a_vals, b_vals, n = align_scores(scores_a, scores_b)
    if n == 0:
        print(f"  {label_a} vs {label_b}: NO OVERLAPPING IDs - cannot compute QWK")
        return None
    kappa = cohen_kappa_score(a_vals, b_vals, weights="quadratic")
    print(f"  {label_a} vs {label_b}: QWK={kappa:.4f}  (N={n})")
    return kappa


def compute_qwk_report(my_scores, yishun_scores, judge_scores):
    print(f"\n{'='*70}")
    print(f"  QWK REPORT: inter-annotator agreement vs. judge agreement")
    print(f"{'='*70}")

    print("\nHuman-human agreement (genuine inter-annotator reliability):")
    human_human = compute_qwk(my_scores, yishun_scores, "You", "Yishun")

    print("\nHuman-AI agreement (judge vs. each human):")
    judge_vs_you = compute_qwk(my_scores, judge_scores, "You", "Judge")
    judge_vs_yishun = compute_qwk(yishun_scores, judge_scores, "Yishun", "Judge")

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION")
    print(f"{'='*70}")
    if human_human is not None and judge_vs_you is not None:
        diff = human_human - judge_vs_you
        print(f"  Human-human QWK ({human_human:.4f}) vs. Judge-You QWK ({judge_vs_you:.4f})")
        if abs(diff) < 0.05:
            print(f"  -> Judge agrees with you about as well as Yishun does. Strong")
            print(f"     validation: the judge is roughly as reliable as a second human.")
        elif diff > 0:
            print(f"  -> Human-human agreement is higher by {diff:.4f}. The judge is somewhat")
            print(f"     less reliable than a second human annotator, but still worth")
            print(f"     reporting both numbers for context.")
        else:
            print(f"  -> Judge-You agreement is HIGHER than human-human agreement by "
                  f"{abs(diff):.4f}.")
            print(f"     Worth double-checking this isn't a fluke of a small sample -")
            print(f"     but if real, it's a genuinely strong result for the judge.")


if __name__ == "__main__":
    # smoke test with synthetic data
    import random
    random.seed(42)
    ids = [f"sample_{i}" for i in range(100)]
    true_sev = {i: random.choices([0,1,2,3,4], weights=[30,30,20,15,5])[0] for i in ids}

    def noisy_copy(scores, noise_prob=0.15):
        result = {}
        for k, v in scores.items():
            if random.random() < noise_prob:
                v = max(0, min(4, v + random.choice([-1, 1])))
            result[k] = v
        return result

    my_scores = noisy_copy(true_sev, 0.1)
    yishun_scores = noisy_copy(true_sev, 0.15)
    judge_scores = noisy_copy(true_sev, 0.2)

    compute_qwk_report(my_scores, yishun_scores, judge_scores)


# ── Real data loader, matching the confirmed annotation file schema ────────────

import json


def load_human_severity(path, uid_key="uid", severity_key="human_severity"):
    """Loads {uid: severity} from an annotation file matching the
    confirmed schema (list of records, each with a uid and a
    human_severity field). Adapt uid_key/severity_key if Yishun's file
    uses different field names for the same thing."""
    with open(path) as f:
        data = json.load(f)

    # handle both a bare list and a {"samples": [...]} / {"rows": [...]} wrapper
    records = data if isinstance(data, list) else (
        data.get("samples") or data.get("rows") or data.get("annotations") or []
    )

    scores = {}
    skipped = 0
    for r in records:
        uid = r.get(uid_key)
        severity = r.get(severity_key)
        if uid is None or severity is None:
            skipped += 1
            continue
        scores[uid] = severity

    print(f"  Loaded {len(scores)} scored samples from {path}"
          f"{f' ({skipped} skipped - missing uid or severity)' if skipped else ''}")
    return scores
