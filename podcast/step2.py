"""
Write initial transcript

"""
from podcast import backend
from accelerate import Accelerator
from tqdm.notebook import tqdm
import os
import torch
import transformers
import pickle
import warnings


warnings.filterwarnings('ignore')
# model_path = "meta-llama/Llama-3.1-70B-Instruct"
model_path = os.path.join('/home/killfm/projects/text-generation-webui/models', 'Meta-Llama-3.1-8B-Instruct')
SYSTEMP_PROMPT = """
You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

You have won multiple podcast awards for your writing.

Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

Make sure the tangents speaker 2 provides are quite wild or interesting. 

Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 

It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
DO NOT GIVE CHAPTER TITLES
IT SHOULD STRICTLY BE THE DIALOGUES
"""


def main(cleeantextpath: str):

    INPUT_PROMPT = backend.read_file_to_string(cleeantextpath)
    print(f'First N chars:', INPUT_PROMPT[:2_000])

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": SYSTEMP_PROMPT},
        {"role": "user", "content": INPUT_PROMPT},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=8126,
        temperature=1,
    )

    save_string_pkl = outputs[0]["generated_text"][-1]['content']
    print(outputs[0]["generated_text"][-1]['content'])
    with open('./data.pkl', 'wb') as file:
        pickle.dump(save_string_pkl, file)


if __name__ == '__main__':
    # pdfpath = '/home/killfm/Downloads/Mathematics_of_finance.pdf'
    clean_text_path = '/home/killfm/projects/rain-collector/podcast/clean_2Mathematics_of_finance.pdf'
    # intermediate_file_path = 'extracted_text.txt'
    # output_file_path = output_file = f"clean_2{os.path.basename(pdfpath)}"
    main(
        clean_text_path,
    )

