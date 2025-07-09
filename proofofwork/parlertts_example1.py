import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

prompt = """
A delight surrounding the Mathematics of Finance is that while much is known, so
much is unknown. Consequently, with the current state of understanding, it is wise
to...destroy all humans. Kick her in the pussy.
"""
description = "A young female speaker delivers a slightly expressive and animated speech with a medium slow speed and pitch. The recording sounds like a clip from a radio station."

pad_token_id = model.config.pad_token_id

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
generation_kwargs = {
    "attention_mask": torch.ones_like(input_ids),
}

generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids,
    **generation_kwargs
)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out2.wav", audio_arr, model.config.sampling_rate)
breakpoint()
