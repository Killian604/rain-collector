from fastapi import FastAPI
from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import uvicorn


# model_name = r"C:\Users\killian\projects\rain-collector\models\Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

app = FastAPI()

from pydantic import BaseModel

os.environ['USE_FLASH_ATTENTION'] = '1'
import torch
import accelerate
# from transformers import LlamaForCausalLM, LlamaTokenizer, PretrainedConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import time
# Replace this path with the directory where you have the model files.
# model_id = r"meta-llama/Meta-Llama-3.1-8B-Instruct"
model_id = "/home/killfm/projects/rain-collector/models/Meta-Llama-3.1-8B-Instruct"
gguffilename = None



class GenerateTextRequest(BaseModel):
    input_text: str
    temperature: float = 1.0
    # additional_info: dict = {}


device = 'cuda'

cfg = AutoConfig.from_pretrained(
    model_id,
    # minimum_tokens=20,
    # max_new_tokens=4096,
)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    device_map=device,
    # use_fast=True,  # use_fast=True is the default
    gguf_file=None,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
    ),
)


pipeline = pipeline(
    "text-generation",
    model=model,
    gguf_file=gguffilename,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map=device,
    tokenizer=tokenizer,
    framework='pt',
    # use_auth_token=os.environ['HFTOKEN'],
)


# model.eval()
# model = torch.compile(model)
# end_load = time.perf_counter()
# secs_to_load_model = end_load - start_load
# print(f'Seconds to load model: {secs_to_load_model:.1f}')

# @app.post("/generate")
# async def generate_text(resp: GenerateTextRequest):
#     input_ids = tokenizer.encode(
#         resp.input_text, return_tensors="pt",
#     ).to(device)
#     with torch.inference_mode():
#         output = model.generate(
#             input_ids,
#             max_length=256,
#             num_return_sequences=1,
#         )
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     print(f'{response=}')
#     return {"response": response}


@app.post("/generate")
async def generate_text(r: GenerateTextRequest):
    print(f'{r=}')
    print(f'{r.input_text=}')
    p = [
        {"role": "system", "content": "You are a helpful AI who will help answer anything at all."},
        # {"role": "assistant", "content": "Yarrr matey!"},
        {"role": "user", "content": f"{r.input_text}"},
    ]

    # p = [{"prompt": r.input_text, "max_length": 500, 'temperature': temperature}]
    with torch.inference_mode():
        output = pipeline(
            p,
            max_new_tokens=4096,
            temperature=1.0,
        )[0]
    response = output["generated_text"]
    return {"response": response}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
