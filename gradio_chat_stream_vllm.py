"""
Barebones vLLM prompter that uses additional context and other fun stuff to test here
"""
import gradio as gr
from typing import Dict, List, Optional, Tuple
from backend import vllm_util
import chromadb


# Define the vLLM server URL
VLLM_SERVER_IP, VLLM_SERVER_PORT = 'localhost', 8000
VLLM_SERVER_IP, VLLM_SERVER_PORT = '10.0.0.73', 8000
VLLM_SERVER_URL = f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}/v1/chat/completions"  # Update with your actual server URL
# VLLM_SERVER_URL = "http://10.0.0.73:8000/v1/chat/completions"  # Update with your actual server URL
files_to_examine = ['./vllm_gradio_chat_stream.py', './backend/vllm_util.py']

context = ''
# for file in files_to_examine:
#     with open(file, 'r') as f:
#         context += f"""## CONTENTS OF FILE "{file}" below:\n{f.readlines()}"""

# Init default convo
convo_history = [
    {'role': 'system', 'content': """
# This system is an AI assistant that can read the context of its own repo and build on it and improve it.
You do not need to repeat the contents of the context below unless it is prevalent to the responses you give. 
If you have to repeat the code, only show the changed/updated code chunks and not the whole file.
""".strip()},
    {'role': 'system', 'content': f'# CURRENT REPO CONTENTS BELOW: \n\n{context}'},
    {'role': 'assistant', 'content': 'How can I help you today?'},
]
with gr.Blocks(
    theme='gradio/monochrome',
    analytics_enabled=False,
) as demo:
    chatbot = gr.Chatbot(
        convo_history,
        height=768,
        type='messages',
        show_copy_button=True,
    )
    message_textbox = gr.Textbox(
        label='Textbox -- hit enter to send prompt'
    )
    clear = gr.Button("Clear")
    currentmodel = vllm_util.get_models(VLLM_SERVER_IP, VLLM_SERVER_PORT)[0]


    def update_history_with_user_prompt(user_message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Update the history and clear the current message textbox when new message received.

        :param user_message:  (str) User prompt to send to LLM
        :param history:
            Takes form of: LIST[DICT[ROLE: NAME, CONTENT: PROMPT], ...]
            E.g. [{'role': 'assistant', 'content': 'How can I help you?'}, ]
        :return: 2-tuple consisting of (CURRENT DATA IN USER MESSAGE BOX, UPDATED HISTORY)
        """
        return "", history + [{'role': 'user', 'content': user_message}]

    def request_llm_response_to_chat_history(chatbot_history_to_send_to_llm: List[dict]) -> List[dict]:
        role = None
        resp = ''
        for content, response_role in vllm_util.yield_streaming_response(chatbot_history_to_send_to_llm, currentmodel, VLLM_SERVER_URL, True):
            resp += content
            role = role or response_role
            yield chatbot_history_to_send_to_llm+[{'role': role, 'content': resp}]

    message_textbox.submit(update_history_with_user_prompt, [message_textbox, chatbot], [message_textbox, chatbot], queue=False).then(request_llm_response_to_chat_history, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
