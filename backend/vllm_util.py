"""

"""
from typing import Iterable, List, Tuple, Union
import argparse
import json
import requests

DEFAULT_MAX_TOKENS = 1024


# General
def render_model_api_uri(host, port):
    return f"http://{host}:{port}/v1/models"


# Model listing
def _get_models_response(host, port) -> requests.Response:
    """

    Response.json() looks like:
        j={'object': 'list', 'data': [{'id': './NousResearch_Hermes-3-Llama-3.1-8B/', 'object': 'model', 'created': 1726188329, 'owned_by': 'vllm', 'root': './NousResearch_Hermes-3-Llama-3.1-8B/', 'parent': None, 'max_model_len': 40000, 'permission': [{'id': 'modelperm-f6e6486441fb4ee7adc4c5c3bea81b45', 'object': 'model_permission', 'created': 1726188329, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]}
        modelsnamesdicts=[{'id': './NousResearch_Hermes-3-Llama-3.1-8B/', 'object': 'model', 'created': 1726188329, 'owned_by': 'vllm', 'root': './NousResearch_Hermes-3-Llama-3.1-8B/', 'parent': None, 'max_model_len': 40000, 'permission': [{'id': 'modelperm-f6e6486441fb4ee7adc4c5c3bea81b45', 'object': 'model_permission', 'created': 1726188329, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]
    """
    api_url = render_model_api_uri(host, port)
    headers = {"User-Agent": "Test Client"}
    try:
        response = requests.get(
            api_url,
            headers=headers,
            timeout=50,
        )
    except requests.exceptions.ConnectionError as e:
        print(f'Bad URI: {api_url} Check: {repr(e)}')
        raise e

    return response


def _parse_models_resp_for_model_names(r: requests.Response) -> List[str]:
    """

    :param r:
    :return:
    Example return JSON #1:
        {'object': 'list',
         'data': [
            {'id': './NousResearch_Hermes-3-Llama-3.1-8B/',
            'object': 'model',
            'created': 1726188329,
            'owned_by': 'vllm',
            'root': './NousResearch_Hermes-3-Llama-3.1-8B/',
            'parent': None,
            'max_model_len': 40000,
            'permission': [
                {'id': 'modelperm-f6e6486441fb4ee7adc4c5c3bea81b45',
                'object': 'model_permission',
                'created': 1726188329,
                'allow_create_engine': False,
                'allow_sampling': True,
                'allow_logprobs': True,
                'allow_search_indices': False,
                'allow_view': True,
                'allow_fine_tuning': False,
                'organization': '*',
                'group': None,
                'is_blocking': False}]}]}
    """
    if r.status_code != 200:
        raise ValueError(f'Unusual status code/resp: {r.status_code=}')

    j = r.json()
    if 'data' not in j:
        raise ValueError(f'Missing expected key in JSON "data": {j=}')
    return [x['id'] for x in j['data']]


def get_models(host: str, port: Union[str, int]):
    """

    :param host:
        E.g. 'localhost'
        E.g. '127.0.0.1'
    :param port:
        E.g. 8000
    :return:
    """
    return _parse_models_resp_for_model_names(_get_models_response(host, port))


# Unsorted
def post_http_prompt_request(
        prompt: str,
        modelname: str,
        api_url: str,
        temperature: float = 0.0,
        n: int = 1,
        stream: bool = False,
        max_tokens=1024,
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        'model': modelname,
        "prompt": prompt,
        "n": n,
        "use_beam_search": n > 1,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(
        api_url,
        headers=headers,
        json=pload,
        stream=stream,
    )
    return response


def post_chat_request(
        chat: List[dict],
        model: str,
        api_url: str,
        n: int = 1,
        stream: bool = False,
        max_tokens=DEFAULT_MAX_TOKENS,
) -> requests.Response:
    """

    :param chat:
        E.g. [{'role':'system', 'content': 'You are an AI assistant.'}, {'role':'user', 'content': 'What is the largest city on Earth?'}]

    :param model:
    :param api_url:
    :param n:
    :param stream:
    :param max_tokens:
    :return:
    """
    print(f'Post chat request called')
    headers = {"User-Agent": "Test Client"}
    pload = {
        'model': model,
        "messages": chat,
        "n": n,
        "use_beam_search": n > 1,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    response = requests.post(
        api_url,
        headers=headers,
        json=pload,
        stream=stream,
    )
    # print(f'{response.status_code}')
    return response


# def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
#     for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
#         if chunk:
#             data = json.loads(chunk.decode("utf-8"))
#             output = data['choices']
#             yield output


def prompt_response(response: requests.Response) -> List[str]:
    if response.status_code == 200:
        data = json.loads(response.content)
        print(f'{data=}')
        output = data['choices']
    else:
        output = f'{response.status_code=} // {json.loads(response.content)=}'
    return output


def parse_chat_response(response) -> Tuple[str, str]:
    """

    :param response:
    :return:

    Example JSON:
        {'id': 'chat-523be9a6378047fea1e4123f14b9ab8c',
        'object': 'chat.completion',
        'created': 1726339186,
        'model': './NousResearch_Hermes-3-Llama-3.1-8B/',
        'choices': [{'index': 0,
        'message': {'role': 'assistant',
            'content': 'The largest city on Earth, in',
            'tool_calls': []},
        'logprobs': None,
        'finish_reason': 'length',
        'stop_reason': None}],
    'usage': {'prompt_tokens': 28, 'total_tokens': 35, 'completion_tokens': 7},
    'prompt_logprobs': None}
    """
    j = response.json()
    return j['choices'][0]['message']['content'], j['choices'][0]['message']['role']

def parse_chat_response_stream(resp, debug=True) -> Tuple[str, str]:
    """

    :param resp:
    :return:
    """
    role = 'UNKNOWN'
    final_out = ''
    print(f'parse_chat_response_stream CALLED! {resp=}')
    for i, chunk in enumerate(resp.iter_lines(
            chunk_size=8192,
            decode_unicode=False,
            # delimiter=b"data:",
            delimiter=b"data:",
    )):
        # print(f'{i=}')
        chunk_decoded: str = chunk.decode('utf-8')
        if chunk_decoded:
            if debug:
                print(f'{i=} {chunk_decoded=}')
            chunk_decoded_dataremoved = chunk_decoded.strip()  # .removeprefix('data: ')
            """Example chunk after JSON parse: {"id":"chat-281fefb9951f44e3a24635359ff6b4c7","object":"chat.completion.chunk","created":1726770376,"model":"/home/killfm/projects/text-generation-webui/models/NousResearch_Hermes-3-Llama-3.1-8B/","choices":[{"index":0,"delta":{"content":" area"},"logprobs":null,"finish_reason":null}]}'"""
            if '[DONE]' in chunk_decoded_dataremoved or '<|end_of_text|>' in chunk_decoded:
                print(f'DONE detected or EOT')
                return final_out, role

            try:
                data: dict = json.loads(chunk_decoded_dataremoved)
            except BaseException as e:
                err = f'Err: {repr(e)=} // {chunk_decoded_dataremoved=}'
                print(err)
                breakpoint()
            #     continue
                # raise StopIteration
                # print(f'{chunk_decoded=}')
                # raise
            if debug:
                print(f'{data=}')
            output = data['choices'][0]['delta']['content']
            role = data['choices'][0]['delta'].get('role') or role
            # role = data['choices'][0]['delta']['role']
            final_out += output
            if debug:
                print(f'{output=}')
            # breakpoint()
            # print('---Done one---')
            yield output, role
            # if '[DONE]' in chunk:
            #     raise StopIteration


def http_bot(prompt, llm_uri, modelname, n=1, max_tokens=500,stream=True):
    final_text = ''
    print(f'{prompt=}')
    headers = {"User-Agent": "vLLM Client"}
    pload = {
        'model': modelname,
        "prompt": str(prompt),
        "n": n,
        "use_beam_search": n > 1,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
        # "logprobs": 1,
    }
    response = requests.post(llm_uri, headers=headers, json=pload, stream=True, )
    return parse_chat_response_stream(response)
    for i, chunk in enumerate(response.iter_lines(
            chunk_size=8192,
            decode_unicode=False,
            delimiter=b"data:",
    )):
        chunk_decoded: str = chunk.decode('utf-8')
        if chunk_decoded:
            # if not chunk_decoded.startswith('data: '):
            #     raise Exception(f'Unexpected format for {chunk_decoded=}')
            print(f'{i=} {chunk_decoded=}')
            chunk_decoded_dataremoved = chunk_decoded.strip()  # .removeprefix('data: ')
            if '[DONE]' in chunk_decoded_dataremoved or '<|end_of_text|>' in chunk_decoded:
                return

            try:
                data: dict = json.loads(chunk_decoded_dataremoved)
            except BaseException as e:
                err = f'Err: {repr(e)=} // {chunk_decoded_dataremoved=}'
                print(err)
                # breakpoint()
            #     continue
                # raise StopIteration
                # print(f'{chunk_decoded=}')
                # raise
            # print(f'{data=}')
            output = data['choices'][0]['text']
            final_text += output
            print(f'{output=}')
            # print('---Done one---')
            yield final_text
            # if '[DONE]' in chunk:
            #     raise StopIteration


# Primary funcs


def yield_resp(chat, model, api_url, stream: bool, max_tokens=DEFAULT_MAX_TOKENS):
    print(f'yiueldresp cllaed')
    response = post_chat_request(
        chat,
        model,
        api_url,
        stream=stream,
        max_tokens=max_tokens,
    )
    if stream:
        for c in parse_chat_response_stream(response):
            yield c
    else:
        yield parse_chat_response(response)