"""

Dev note: referencing a specific local model is approx. 3x faster on total execution than asking HF to search cache.
"""
from colorama import init, Fore, Style
from ply.yacc import token
from transformers import AutoTokenizer  # from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import time
# os.environ['USE_FLASH_ATTENTION'] = '1'


if __name__ == '__main__':
    start = time.perf_counter()
    init()

    # Replace this path with the directory where you have the model files.
    # model_id = r"meta-llama/Meta-Llama-3.1-8B-Instruct"  # This phrasing would check cache
    model_id = "/home/killfm/projects/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct"

    mytext = 'Default text goes here!'
    mytext = 'You can also use ANSI escape codes to add color to your STDOUT output without using the colorama library. Here\'s an example'
    # assert os.path.isdir(model_id), f'Dir not found: {model_id=}'

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        device_map='cpu',
    )

    inputtext = mytext if len(sys.argv) <= 1 else ' '.join(sys.argv[1:])

    tokenizer_output = tokenizer(inputtext)
    total_tokens = len(tokenizer_output["input_ids"])
    print(f'Input text: {inputtext}')
    print(f'{tokenizer_output=}')
    print(Style.BRIGHT + Fore.GREEN + f'Number of tokens for input text: {total_tokens - 1}'.rjust(25) + Style.RESET_ALL)
    print(Style.BRIGHT + Fore.GREEN + f'{total_tokens - 1} text tokens generated'.rjust(25) + Style.RESET_ALL)
    print(f'(Total tokens generated: {total_tokens})'.rjust(25))

    print(f'\n--- Start of per-word analysis---')
    for i, word in enumerate(inputtext.split()):
        tokenoutput = tokenizer(word)['input_ids']
        tokenoutputtextonly = tokenoutput[1:]

        print(f'{i:02d}: Input text: {word.rjust(22)} / Tokens={len(tokenoutputtextonly)}')
        for v in tokenoutputtextonly:
            iv_str = tokenizer.decode(v)
            print(f'\t{iv_str=}')
        print()

    print()
    secs_to_process = time.perf_counter() - start

    print(f'Secs to process request: {secs_to_process:.2f}')
