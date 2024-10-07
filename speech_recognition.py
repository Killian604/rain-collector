"""

"""
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import torch
import os
os.environ['TORCHDYNAMO_VERBOSE'] = '1'


mp3path = '/home/killfm/Downloads/cbtlessons178b43a886a10a7c89cecabaebd6eb61750c5cd74.m4a'
mp3path = '/home/killfm/Downloads/jreAJ.mp3'
# mp3path = f'/home/killfm/Downloads/motoko1.mp3'
assert os.path.isfile(mp3path)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'{device=}')
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    se_safetensors=True,
)
model.to(device)

# # Enable static cache and compile the forward pass
# model.generation_config.cache_implementation = "static"
# model.generation_config.max_new_tokens = 256
# model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

processor = AutoProcessor.from_pretrained(model_id)
time_start = time.perf_counter()

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True
)

result = pipe(mp3path)
time_end = time.perf_counter()
elapsed_secs = time_end - time_start
# print(f'{result=}')
print(f'Time to process transcription: {elapsed_secs:.2f=}')
print(f'{result["text"]}')
breakpoint()
