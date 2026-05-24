import os
from openai import OpenAI
from dotenv import load_dotenv

hyp = "the subject fled the scene"
ref = "the suspect fled the scene"
load_dotenv()
MODEL = "gpt-4o"
api_key=os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
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

        Reply with only: true or false"""},
        {"role": "user", "content": f"Reference: {ref}\nHypothesis: {hyp}"}])
print(hyp)
print(ref)
print("Assistant: "+ completion.choices[0].message.content)
