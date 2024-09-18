"""

"""
from typing import Iterable, List, Tuple
import argparse
import json
import requests

def render_model_api_uri(host, port):
    return f"http://{host}:{port}/v1/models"

def post_http_prompt_request(
        prompt: str,
        modelname: str,
        api_url: str,
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
    return response



def post_http_chat_request(
        chat: List[dict],
        modelname: str,
        api_url: str,
        n: int = 1,
        stream: bool = False,
        max_tokens=1024,
) -> requests.Response:
    """

    :param chat:
        E.g. [{'role':'system', 'content': 'You are an AI assistant.'}, {'role':'user', 'content': 'What is the largest city on Earth?'}]

    :param modelname:
    :param api_url:
    :param n:
    :param stream:
    :param max_tokens:
    :return:
    """
    headers = {"User-Agent": "Test Client"}
    pload = {
        'model': modelname,
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
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data['choices']
            yield output


def prompt_response(response: requests.Response) -> List[str]:
    if response.status_code == 200:
        data = json.loads(response.content)
        print(f'{data=}')
        output = data['choices']
    else:
        output = f'{response.status_code=} // {json.loads(response.content)=}'
    return output



def _get_models_response(host, port) -> requests.Response:
    """



    :param prompt:
    :param api_url:
    :param n:
    :param stream:
    :param max_tokens:
    :return:

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


def parse_models_resp_for_model_names(r: requests.Response) -> List[str]:
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

def get_models(host, port):
    """

    :param host:
        E.g. 'localhost'
        E.g. '127.0.0.1'
    :param port:
        E.g. 8000
    :return:
    """
    return parse_models_resp_for_model_names(_get_models_response(host, port))


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

