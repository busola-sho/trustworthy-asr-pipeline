from jiwer import wer
from src.models import ASRModel
from src.datasets import Dataset
import json


def run_benchmark(model: ASRModel, dataset: Dataset, output_path: str, max_samples: int = None) -> dict:
    all_refs=[]
    all_hyps=[]
    results=[]

    for i, sample in enumerate(dataset.load()):
        if max_samples and i >= max_samples:
            break
        transcript=model.transcribe(sample.audio, sample.sample_rate)
        sample_wer=wer(sample.label.lower(),transcript.text.lower())
        results.append(
            {
                "ref": sample.label,
                "hyp": transcript.text,
                "sample_WER": sample_wer,
            }
        )
        all_refs.append(sample.label.lower())
        all_hyps.append(transcript.text.lower())

    corpus_wer = wer(all_refs,all_hyps)
    count= len(all_hyps) if len(all_hyps)==len(all_refs) else -1

    output = {
        "model": model.model_name,
        "dataset": dataset.name,
        "corpus_wer": corpus_wer,
        "num_samples": count,
        "samples": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return {"model":model.model_name, "dataset":dataset.name, "num_of_samples":count, "WER":corpus_wer}

