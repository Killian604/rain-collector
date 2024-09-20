import gradio as gr
import random
import time
import gradio as gr
import requests
from typing import Dict, List, Optional, Tuple
from backend import vllm_util
import os
import uvicorn
from uvicorn.config import LOGGING_CONFIG

# Define the vLLM server URL
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"  # Update with your actual server URL

convo_history = [
    {'role': 'system', 'content': 'You are an AI assistant..'},
    # {'role': 'system', 'content': f'The following are the contents of "vllm_util.py":\n\n{contents}'},
    {'role': 'assistant', 'content': 'How can I help you today?'},
    # {'role': 'user', 'content': prompt},
]
with gr.Blocks(
    theme='gradio/monochrome',
    analytics_enabled=False,
) as demo:
    chatbot = gr.Chatbot(
        convo_history,
        height=1024,
        type='messages',
        show_copy_button=True,
    )
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    currentmodel = vllm_util.get_models('localhost', 8000)[0]


    def user(user_message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """

        :param user_message:  (str) User prompt to send to LLM
        :param history:
            Takes form: LIST[DICT[ROLE: NAME, CONTENT: PROMPT], ...]
            E.g. [{'role': 'assistant', 'content': 'How can I help you?'}]

        :return: 2-tuple consisting of (CURRENT DATA IN USER MESSAGE BOX, UPDATED HISTORY)
        """
        print(f'{history=}')
        return "", history + [{'role': 'user', 'content': user_message}]


    def bot(chatbot_history_to_send_to_llm: List[dict]) -> List[dict]:
        # print(f'bot(): {chatbot_history_to_send_to_llm=}')
        # chatbot_history_cleaned = [x for x in chatbot_history]
        role = None
        resp = ''
        for content, response_role in vllm_util.yield_streaming_response(chatbot_history_to_send_to_llm, currentmodel, VLLM_SERVER_URL, True):
            resp += content
            role = role or response_role
            yield chatbot_history_to_send_to_llm+[{'role': role, 'content': resp}]

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
