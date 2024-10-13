"""

"""
from transformers import AutoTokenizer  # from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import time
# os.environ['USE_FLASH_ATTENTION'] = '1'


if __name__ == '__main__':
    start = time.perf_counter()
    # Replace this path with the directory where you have the model files.
    # model_id = r"meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = "/home/killfm/projects/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct"
    mytext = 'Thanks'


    # assert os.path.isdir(model_id), f'Dir not found: {model_id=}'

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map='cpu',
        # use_fast=True,  # use_fast=True is the default
        # gguf_file=None,
    )

    inputtext = mytext if len(sys.argv) <= 1 else ' '.join(sys.argv[1:])

    output = tokenizer(inputtext)
    secs_to_process = time.perf_counter() - start
    total_tokens = len(output["input_ids"])
    print(f'Input text: {inputtext}')
    print(f'{output=}')
    print(f'Number of tokens for input text: {total_tokens-1}')
    print(f'(Total tokens generated: {total_tokens})')
    print(f'Secs to process request: {secs_to_process:.2f}')
