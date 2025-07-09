from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import uvicorn


# model_name = r"C:\Users\killian\projects\rain-collector\models\Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
app = FastAPI()
os.environ['USE_FLASH_ATTENTION'] = '1'
import torch
import accelerate
# from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import time
# Replace this path with the directory where you have the model files.
# model_id = r"meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "/home/killfm/projects/rain-collector/models/Meta-Llama-3.1-8B-Instruct"
device = 'cuda'


from pydantic import BaseModel

class GenerateTextRequest(BaseModel):
    input_text: str
    temperature: float = 1.0
    additional_info: dict = {}



cfg = AutoConfig.from_pretrained(
    model_id,
    minimum_tokens=20,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    device_map=device,

    # use_fast=True,  # use_fast=True is the default
    # gguf_file=None,
)

# Load the model
start_load = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
    ),
)
# model.eval()
model = torch.compile(model)
end_load = time.perf_counter()
secs_to_load_model = end_load - start_load
print(f'Seconds to load model: {secs_to_load_model:.1f}')

@app.post("/generate")
async def generate_text(resp: GenerateTextRequest):
    input_ids = tokenizer.encode(
        resp.input_text, return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_length=256,
            num_return_sequences=1,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'{response=}')
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
