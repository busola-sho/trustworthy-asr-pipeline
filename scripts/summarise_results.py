import json
import glob

def summarise():
    files = glob.glob("benchmarks/*.json")
    
    results = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)
        
        if "corpus_wer" not in data:
            continue
        
        results.append({
            "model": data.get("model", "unknown").split("/")[-1],
            "dataset": data.get("dataset", "unknown"),
            "num_samples": data.get("num_samples", 0),
            "wer": round(data.get("corpus_wer", 0) * 100, 2),
            "mar": round(data.get("meaning_alteration_rate", 0) * 100, 2),
        })
    
    datasets = {}
    for r in results:
        datasets.setdefault(r["dataset"], []).append(r)
    
    md = ""
    for dataset_name, rows in sorted(datasets.items()):
        best = min(rows, key=lambda x: x["wer"])
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        print(f"{'Model':<35} {'Samples':<10} {'WER %':<10} {'MAR %':<10}")
        print("-" * 65)
        for r in sorted(rows, key=lambda x: x["wer"]):
            flag = " <- best" if r["model"] == best["model"] else ""
            print(f"{r['model']:<35} {r['num_samples']:<10} {r['wer']:<10} {r['mar']:<10}{flag}")
        
        md += f"\n### {dataset_name}\n\n"
        md += "| Model | Samples | WER % | MAR % |\n"
        md += "|-------|---------|-------|-------|\n"
        for r in sorted(rows, key=lambda x: x["wer"]):
            if r["model"] == best["model"]:
                md += f"| **{r['model']}** | {r['num_samples']} | **{r['wer']}** | {r['mar']} |\n"
            else:
                md += f"| {r['model']} | {r['num_samples']} | {r['wer']} | {r['mar']} |\n"
        md += "\n"
    
    with open("benchmarks/summary.md", "w") as f:
        f.write(md)
    
    print("\nMarkdown saved to benchmarks/summary.md")

summarise()