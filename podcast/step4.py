"""
Write initial transcript

"""
from pydub import AudioSegment
import io
from scipy.io import wavfile
from IPython.display import Audio
import IPython.display as ipd
from tqdm import tqdm
import os
import transformers
import pickle
from tqdm.notebook import tqdm
import warnings
from podcast import backend
# Import necessary libraries
from accelerate import Accelerator
from transformers import BarkModel, AutoProcessor, AutoTokenizer
import torch
import json
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration
from tqdm.notebook import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# model_path = "meta-llama/Llama-3.1-70B-Instruct"
model_path = os.path.join('/home/killfm/projects/text-generation-webui/models', 'Meta-Llama-3.1-8B-Instruct')


def poc():
    # Set up device
    device = "cuda"  # if torch.cuda.is_available() else "cpu"


    # # # PARLER
    # Load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    # Define text and description
    text_prompt = """
    Exactly! And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
    """
    description = """
    Laura's voice is expressive and dramatic in delivery, speaking at a fast pace with a very close recording that almost has no background noise.
    """
    # Tokenize inputs
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text_prompt, return_tensors="pt").input_ids.to(device)
    print('Audio tokenized')
    # Generate audio
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    print(f'Audio generated')

    pass
    # wavfile.write('test.wav', 44100, audio_arr)

    # Play audio in notebook
    ipd.Audio(audio_arr, rate=model.config.sampling_rate)


    # BARK
    voice_preset = "v2/en_speaker_6"
    sampling_rate = 24000
    device = "cuda"
    processor = AutoProcessor.from_pretrained("suno/bark")
    # model =  model.to_bettertransformer()
    # model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
    model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)  # .to_bettertransformer()
    text_prompt = """
    Exactly! [sigh] And the distillation part is where you take a LARGE-model,and compress-it down into a smaller, more efficient model that can run on devices with limited resources.
    """
    inputs = processor(text_prompt, voice_preset=voice_preset).to(device)

    speech_output = model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    from scipy.io import wavfile
    wavfile.write('test3.wav', 24000, speech_output.cpu().numpy().astype(np.float32).squeeze())




def mainmain():
    device = 'cuda'

    with open('./podcast_ready_data.pkl', 'rb') as file:
        PODCAST_TEXT = pickle.load(file)
    bark_processor = AutoProcessor.from_pretrained("suno/bark")
    bark_model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to(device)
    bark_sampling_rate = 24000

    parler_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    speaker1_description = """
    Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
    """
    generated_segments = []
    sampling_rates = []  # We'll need to keep track of sampling rates for each segment

    def generate_speaker1_audio(text):
        """Generate audio using ParlerTTS for Speaker 1"""
        input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        return audio_arr, parler_model.config.sampling_rate

    def generate_speaker2_audio(text):
        """Generate audio using Bark for Speaker 2"""
        inputs = bark_processor(text, voice_preset="v2/en_speaker_6").to(device)
        speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
        audio_arr = speech_output[0].cpu().numpy()
        return audio_arr, bark_sampling_rate

    def numpy_to_audio_segment(audio_arr, sampling_rate):
        """Convert numpy array to AudioSegment"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_arr * 32767).astype(np.int16)

        # Create WAV file in memory
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sampling_rate, audio_int16)
        byte_io.seek(0)

        # Convert to AudioSegment
        return AudioSegment.from_wav(byte_io)

    import ast
    ast.literal_eval(PODCAST_TEXT)
    final_audio = None

    i=0
    for speaker, text in tqdm(ast.literal_eval(PODCAST_TEXT), desc="Generating podcast segments", unit="segment"):
        i += 1
        if speaker == "Speaker 1":
            audio_arr, rate = generate_speaker1_audio(text)
            # print(f'Done processing text: {text:100}')
        else:  # Speaker 2
            audio_arr, rate = generate_speaker2_audio(text)
        print(f'Done processing text: {text:100}')

        # Convert to AudioSegment (pydub will handle sample rate conversion automatically)
        audio_segment = numpy_to_audio_segment(audio_arr, rate)

        # Add to final audio
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment
        if i >= 5:
            break
    final_audio.export("_podcast.mp3",
                       format="mp3",
                       bitrate="192k",
                       parameters=["-q:a", "0"])

    pass

if __name__ == '__main__':
    # pdfpath = '/home/killfm/Downloads/Mathematics_of_finance.pdf'
    # clean_text_path = '/home/killfm/projects/rain-collector/podcast/clean_2Mathematics_of_finance.pdf'
    # intermediate_file_path = 'extracted_text.txt'
    # output_file_path = output_file = f"clean_2{os.path.basename(pdfpath)}"
    # pklpath = './data.pkl'
    mainmain(

    )

