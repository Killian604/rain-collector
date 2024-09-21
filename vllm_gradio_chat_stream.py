"""

"""
import gradio as gr
from typing import Dict, List, Optional, Tuple
from backend import vllm_util

# Define the vLLM server URL
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"  # Update with your actual server URL
files_to_examine = ['./vllm_gradio_chat_stream.py', './backend/vllm_util.py']
context = ''
for file in files_to_examine:
    with open(file, 'r') as f:
        context += f"""CONTENTS OF FILE "{file}" below:\n{f.readlines()}"""
convo_history = [
    {'role': 'system', 'content': """
This system is an AI assistant that can read the context of its own repo and build on it and improve it.
You do not need to repeat the contents of the context below to the user unless it is prevalent to the responses you give.
""".replace('\n', ' ').strip()},
    {'role': 'system', 'content': f'CURRENT REPO CONTENTS BELOW: \n\n{context}'},
    {'role': 'assistant', 'content': 'How can I help you today?'},
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


    def update_history_with_user_prompt(user_message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """

        :param user_message:  (str) User prompt to send to LLM
        :param history:
            Takes form: LIST[DICT[ROLE: NAME, CONTENT: PROMPT], ...]
            E.g. [{'role': 'assistant', 'content': 'How can I help you?'}]

        :return: 2-tuple consisting of (CURRENT DATA IN USER MESSAGE BOX, UPDATED HISTORY)
        """
        return "", history + [{'role': 'user', 'content': user_message}]


    def bot(chatbot_history_to_send_to_llm: List[dict]) -> List[dict]:
        role = None
        resp = ''
        for content, response_role in vllm_util.yield_streaming_response(chatbot_history_to_send_to_llm, currentmodel, VLLM_SERVER_URL, True):
            resp += content
            role = role or response_role
            yield chatbot_history_to_send_to_llm+[{'role': role, 'content': resp}]

    msg.submit(update_history_with_user_prompt, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
