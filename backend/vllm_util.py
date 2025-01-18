"""

"""
from backend import shared
from typing import Iterable, List, Tuple, Union
import argparse
import json
import requests

DEFAULT_MAX_TOKENS = 8192 - 300  # Subtract approx amount on input for 8192 context window


# General
def render_models_api_uri(host, port):
    return f"http://{host}:{port}/v1/models"

def create_vllm_chat_uri(host, port):
    return f"http://{host}:{port}/v1/chat/completions"  # Update with your actual server URL


# Model listing
def _get_models_response(host, port) -> requests.Response:
    """

    Response.json() looks like:
        j={'object': 'list', 'data': [{'id': './NousResearch_Hermes-3-Llama-3.1-8B/', 'object': 'model', 'created': 1726188329, 'owned_by': 'vllm', 'root': './NousResearch_Hermes-3-Llama-3.1-8B/', 'parent': None, 'max_model_len': 40000, 'permission': [{'id': 'modelperm-f6e6486441fb4ee7adc4c5c3bea81b45', 'object': 'model_permission', 'created': 1726188329, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]}
        modelsnamesdicts=[{'id': './NousResearch_Hermes-3-Llama-3.1-8B/', 'object': 'model', 'created': 1726188329, 'owned_by': 'vllm', 'root': './NousResearch_Hermes-3-Llama-3.1-8B/', 'parent': None, 'max_model_len': 40000, 'permission': [{'id': 'modelperm-f6e6486441fb4ee7adc4c5c3bea81b45', 'object': 'model_permission', 'created': 1726188329, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]
    """
    api_url = render_models_api_uri(host, port)
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
    Example response JSON #1:
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
    Ask a vLLM server which models are available. Returns a list of strings of available models.
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
        model: str,
        api_url: str,
        temperature: float = 0.0,
        n: int = 1,
        stream: bool = False,
        max_tokens=DEFAULT_MAX_TOKENS,
) -> requests.Response:
    """
    Send POST request to vLLM for a single prompt. Return the Response object.
    :param prompt:
    :param model:
    :param api_url:
    :param temperature:
    :param n:
    :param stream:
    :param max_tokens:
    :return:
    """
    headers = {"User-Agent": "Test Client"}
    pload = {
        'model': model,
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
        temperature: float = 1.0,
        debug: bool = False,
) -> requests.Response:
    """

    :param chat:
        E.g. [{'role':'system', 'content': 'You are an AI assistant.'}, {'role':'user', 'content': 'What is the largest city on Earth?'}]
        Dev note: sometimes extraneous keys can be present. This function will automatically clean the contents
        of the chat without mutation.
    :param model: Name of model name to use at vLLM server
    :param api_url: API to vLLM server
    :param n: number of beam responses
    :param stream: (bool) Return streaming response
    :param max_tokens:
    :param temperature: temperature of prompt
    :param debug: (bool) print debug info to console
    :return:
    """

    cleanedchat = [{k:v for k,v in x.items() if k in {'role', 'content'}} for x in chat]
    if debug:
        print(f'Post chat request called')
        print(f'{cleanedchat=}')
    headers = {"User-Agent": "Test Client"}
    pload = {
        'model': model,
        "messages": cleanedchat,
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
    # print(f'{response.status_code}')
    return response


def parse_prompt_response_for_choices(response: requests.Response) -> List[str]:

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
    :return: tuple of ('content', 'role')

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
    try:
        content, role = j['choices'][0]['message']['content'], j['choices'][0]['message']['role']
    except KeyError:
        print(f'{j=}')
        raise
    return content, role


def parse_chat_response_stream(r: requests.Response, debug=False) -> Tuple[str, str]:
    """

    :param r:
    :param debug:
    :return:
    """
    role = 'UNKNOWN'
    final_out = ''
    if debug:
        print(f'parse_chat_response_stream CALLED! {r=}')
    for i, chunk in enumerate(r.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"data:",)):
        # print(f'{i=}')
        chunk_decoded: str = chunk.decode('utf-8')
        if chunk_decoded:

            chunk_decoded_dataremoved = chunk_decoded.strip()
            if debug:
                print(f'{i=} {chunk_decoded_dataremoved=}')
            """Example chunk after JSON parse: {"id":"chat-281fefb9951f44e3a24635359ff6b4c7","object":"chat.completion.chunk","created":1726770376,"model":"/home/killfm/projects/text-generation-webui/models/NousResearch_Hermes-3-Llama-3.1-8B/","choices":[{"index":0,"delta":{"content":" area"},"logprobs":null,"finish_reason":null}]}'"""
            if '[DONE]' in chunk_decoded_dataremoved or '<|end_of_text|>' in chunk_decoded:
                # print(f'DONE detected or EOT')
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
            # methodnotallowed
            if 'choices' not in data:

                raise ValueError(f'Not "choices" available: {data=} \n\n {json.dumps(data, indent=4)}')
            output = data['choices'][0]['delta']['content']
            role = data['choices'][0]['delta'].get('role') or role
            # role = data['choices'][0]['delta']['role']
            final_out += output
            if debug:
                print(f'{output=}')
            yield output, role


def http_bot(prompt: str, llm_uri: str, modelname, n=1, max_tokens=500,stream=True):
    # print(f'{prompt=}')
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


# Primary funcs

def yield_streaming_response(chat: List[dict], model, api_url, stream: bool, max_tokens=DEFAULT_MAX_TOKENS, temperature: float = 1.0, ) -> Tuple[str, str]:
    """
    Stream an LLM response via vLLM

    :param chat:
        E.g. [{'role': 'assistant', 'content': 'I am a helpful AI.'}, {'role': 'user', 'content': 'What is the largest city on Earth?'}]
    :param model: E.g. 'Llama3.1'
    :param api_url: E.g. 'localhost:8000'
    :param stream:
    :param max_tokens: E.g. 1024
    :return: Returns 2-tuple generator in form of (CONTENT, ROLE)
        E.g. generator of: ('Tokyo, Japan', 'assistant')
    """
    if not isinstance(chat, list):
        raise TypeError(f'Bad type for {type(chat)=} / {chat=}')
    response = post_chat_request(
        chat,
        model,
        api_url,
        stream=stream,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.status_code == 405:  # "Method not allowed"
        raise ValueError(f'Error 405 occurred: {response=}')
    if stream:
        for c in parse_chat_response_stream(response):
            yield c
    else:
        yield parse_chat_response(response)


def generate_response(chat: List[dict], model, api_url, stream: bool, max_tokens=DEFAULT_MAX_TOKENS, temperature: float = 1.0, ) -> Tuple[str, str]:
    try:
        content, role = next(yield_streaming_response(chat=chat, model=model,api_url=api_url, stream=stream, max_tokens=max_tokens, temperature=temperature))
    except StopIteration as e:
        raise ValueError(f'No content to yield')
    return content, role


def generate_random_location():
    content, role = yield_streaming_response(shared.chathistory_yield_location, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, max_tokens=DEFAULT_MAX_TOKENS, stream=False)
    return content.strip()


if __name__ == '__main__' and False:
    host, port = '10.0.0.73', 8000

    model = get_models(host, port)[0]
    api_url = create_vllm_chat_uri(host, port)
    print(f'{model=}')

    sample_chat = [
        {'role': 'system', 'content': 'You are a generalized AI that can answer anything you put your mind to.'},
        {'role': 'assistant', 'content': 'How can I help you today?'},
        {'role': 'user', 'content': 'Tell me about the various biomes present on Earth.'}
    ]
    sample_chat = [
        {'role': 'system', 'content': 'You are a general-purpose AI. You help the user. Answer concisely.'},
        {'role': 'user', 'content': 'Please generate a random location in the United States of America.'},
    ]
    content, role = generate_response(sample_chat, model, api_url, max_tokens=1_000, stream=False)
    print(f'{content=}')