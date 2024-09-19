"""
Fetch and print all model names currently hosted on vllm server
"""
from backend import vllm_util
import argparse
from typing import Dict, Iterable, List


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    host, port = args.host, args.port
    api_url = vllm_util.render_model_api_uri(host, port)
    response = vllm_util._get_models_response(host, port)
    j = response.json()
    modelsnamesdicts: List[Dict] = j.get('data')
    print(f'{response=}')
    print(f'{j=}')
    print(f'{modelsnamesdicts=}')
    ms = vllm_util.get_models(host, port)
    print(f'\n\n\n{ms=}')
    # for i, m in enumerate(modelsnamesdicts):
    #     print(f'{i}: {m}\n\n')

    # breakpoint()
