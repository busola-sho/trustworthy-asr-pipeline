from jiwer import wer
from src.models import ASRModel
from src.datasets import Dataset
import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

MODEL = "gpt-4o"
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = re.sub(r"[^\w\s']", '', text)
    return text

def alteration_eval(hyp:str, ref:str, sample_wer:float) -> bool:
    if sample_wer==0:
        return False
    
    completion = client.chat.completions.create( 
        model=MODEL, 
        messages=[ 
        {"role": "system", "content": """You are evaluating automatic speech recognition transcripts for a policing context.

        You will be given a reference transcript and a hypothesis transcript of the same spoken audio.

        Your task is to determine if the hypothesis contains any meaning-altering errors — that is, errors that would cause a police officer or legal professional to misunderstand what was said.

        Ignore differences in:
        - Capitalisation
        - Punctuation
        - Contractions (e.g. "I've" vs "I have")
        - Dialect variations (e.g. "aboot" vs "about", "didnae" vs "didn't")
        - Filler words

        Flag as meaning-altering only if:
        - A word is substituted with a different word that changes the factual content
        - A word is missing or added that changes who did what
        - A name, place, or number is transcribed incorrectly

        Only flag as meaning-altering if a factual error would mislead a police officer or legal professional. 
        Minor rewording, paraphrasing, or omission of filler words should not be flagged.

        Reply with only: true or false"""},
        {"role": "user", "content": f"Reference: {ref}\nHypothesis: {hyp}"}])
    result = completion.choices[0].message.content.strip().lower()
    return result == "true"

# def run_benchmark(model: ASRModel, dataset: Dataset, output_path: str, max_samples: int = None) -> dict:
#     all_refs=[]
#     all_hyps=[]
#     results=[]
#     meaning_altering_count=0
#     for i, sample in enumerate(dataset.load()):
        
#         if max_samples and i >= max_samples:
#             break
def run_benchmark(model: ASRModel, dataset: Dataset, output_path: str, max_samples: int = None, start_from: int = 0) -> dict:

    if start_from > 0 and os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results = existing.get("samples", [])
        meaning_altering_count = sum(1 for s in results if s.get("meaning_altering"))
        all_refs = [normalise(s["ref"]) for s in results if isinstance(s["ref"], str)]
        all_hyps = [normalise(s["hyp"]) for s in results]
    else:
        results = []
        meaning_altering_count = 0
        all_refs = []
        all_hyps = []

    for i, sample in enumerate(dataset.load()):
        if i < start_from:
            continue
        if max_samples and i >= max_samples:
            break
        transcript=model.transcribe(sample.audio, sample.sample_rate)
        sample_wer = wer(normalise(sample.label), normalise(transcript.text))
        # Call alteration_eval here
        meaning_altering = alteration_eval(transcript.text, sample.label, sample_wer)

        if meaning_altering:
            meaning_altering_count += 1
        
        results.append(
            {
                "ref": sample.label,
                "hyp": transcript.text,
                "meaning_altering": meaning_altering,
                "sample_WER": sample_wer,
            }
        )
        all_refs.append(normalise(sample.label))
        all_hyps.append(normalise(transcript.text))
        # write progress after every sample
        with open(output_path, "w") as f:
            json.dump({"progress": len(results), "samples": results}, f, indent=2)

    corpus_wer = wer(all_refs,all_hyps)
    count= len(all_hyps) if len(all_hyps)==len(all_refs) else -1

    output = {
        "model": model.model_name,
        "dataset": dataset.name,
        "corpus_wer": corpus_wer,
        "num_samples": count,
        "meaning_alteration_rate": meaning_altering_count / len(results) if results else 0,
        "samples": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return {"model":model.model_name, "dataset":dataset.name, "num_of_samples":count, "WER":corpus_wer, "meaning_alteration_rate": meaning_altering_count / len(results) if results else 0}

