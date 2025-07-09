""" Bass comes in after first verse.
`pip install stable-audio-tools`

# ORIGINAL PROMPT: Set up text and timing conditioning
conditioning = [{
    "prompt": "128 BPM tech house drum loop",
    "seconds_start": 0,
    "seconds_total": 30
}]

"""
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os


outpath = 'output_gregchat55.wav'
assert not os.path.isfile(outpath), f'File already exists: {outpath=}'


device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)

# Set up text and timing conditioning
conditioning = [{
    # "prompt": "Reading the movie script from The Matrix",
    "prompt": " 80bpm chanting with Japanese tycho drums, intensity builds",
    "seconds_start": 0,
    "seconds_total": 30,
}]

# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=100,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32_767).to(torch.int16).cpu()
torchaudio.save(outpath, output, sample_rate)

