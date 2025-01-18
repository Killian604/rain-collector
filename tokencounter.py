"""

Dev note: referencing a specific local model is approx. 3x faster on total execution than asking HF to search cache.
"""
from backend import app
from colorama import init, Fore, Style
from ply.yacc import token
from transformers import AutoTokenizer  # from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import time
# os.environ['USE_FLASH_ATTENTION'] = '1'





if __name__ == '__main__':
    app.counttokens(
        ' '.join(sys.argv[1:]),
    )
