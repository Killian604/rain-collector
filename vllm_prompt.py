"""Example Python client for `vllm.entrypoints.api_server`
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""
from backend import vllm_util
import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


# def post_http_request(
#         prompt: str,
#         api_url: str,
#         n: int = 1,
#         stream: bool = False,
#         max_tokens=50,
# ) -> requests.Response:
#     headers = {"User-Agent": "Test Client"}
#     pload = {
#         'model': './NousResearch_Hermes-3-Llama-3.1-8B/',
#         "prompt": prompt,
#         "n": n,
#         "use_beam_search": n > 1,
#         "temperature": 0.0,
#         "max_tokens": max_tokens,
#         "stream": stream,
#     }
#     response = requests.post(
#         api_url,
#         headers=headers,
#         json=pload,
#         stream=stream,
#     )
#     return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            try:
                data = json.loads(chunk.decode("utf-8"))
            except:
                print(f'{response.status_code=} // {response}')
                raise
            output = data['choices']
            yield output


# def get_response(response: requests.Response) -> List[str]:
#     print(f'{response=}')
#     data = json.loads(response.content)
#     print(f'{data=}')
#     output = data['choices'][0]['text']
#
#     # output = data["text"]
#     return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true", default=False)
    args = parser.parse_args()
    host, port = args.host, args.port
    prompt = args.prompt
    api_url = f"http://{host}:{port}/v1/completions/"
    n = args.n
    stream = args.stream
    ms = vllm_util.get_models(host, port)
    print(f'{ms=}')
    modelname = vllm_util.get_models(host, port)[0]
    max_tokens = 7
    print(f"Prompt: {prompt!r}\n", flush=True)
    response = vllm_util.post_http_prompt_request(
        prompt,
        modelname,
        api_url,
        n,
        stream=stream,
        max_tokens=max_tokens,
    )

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = vllm_util.prompt_response(response)
        print(f'{output=}')
        text = output[0]['text']
        print(f'{text=}')
        if n > 1:
            for i, line in enumerate(output):
                print(f"Beam candidate {i}: {line!r}", flush=True)