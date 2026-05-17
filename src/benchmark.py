from jiwer import wer
from src.models import ASRModel
from src.datasets import Dataset
import json

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    return text

def run_benchmark(model: ASRModel, dataset: Dataset, output_path: str, max_samples: int = None) -> dict:
    all_refs=[]
    all_hyps=[]
    results=[]

    for i, sample in enumerate(dataset.load()):
        if max_samples and i >= max_samples:
            break
        transcript=model.transcribe(sample.audio, sample.sample_rate)
        sample_wer = wer(normalise(sample.label), normalise(transcript.text))
        results.append(
            {
                "ref": sample.label,
                "hyp": transcript.text,
                "sample_WER": sample_wer,
            }
        )
        all_refs.append(normalise(sample.label))
        all_hyps.append(normalise(transcript.text))

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

