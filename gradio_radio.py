"""
Barebones vLLM prompter that uses additional context and other fun stuff to test here
"""
import gradio as gr
from typing import Dict, List, Optional, Tuple
from backend import gradio_logic, shared, vllm_util
import chromadb

if gr.NO_RELOAD:
    # from transformers import pipeline
    # pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # TODO: Load audio model
    pass

# Define the vLLM server URL
VLLM_SERVER_IP, VLLM_SERVER_PORT = 'localhost', 8000
VLLM_SERVER_IP, VLLM_SERVER_PORT = '10.0.0.73', 8000

# Save shared
shared.VLLM_SERVER_URL = vllm_util.create_vllm_chat_uri(VLLM_SERVER_IP, VLLM_SERVER_PORT)  # f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}/v1/chat/completions"  # Update with your actual server URL
shared.CURRENT_ML_MODEL = vllm_util.get_models(VLLM_SERVER_IP, VLLM_SERVER_PORT)[0]
# VLLM_SERVER_URL = "http://10.0.0.73:8000/v1/chat/completions"  # Update with your actual server URL

# Init default convo
# TODO: Fill context history
convo_history = [
    {'role': 'system', 'content': """
""".strip()},
    {'role': 'assistant', 'content': 'How can I help you today?'},
]

# Pure functions
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


with gr.Blocks(
        theme='gradio/monochrome',
        analytics_enabled=False,
) as demo:
    state_mostrecent_reply = gr.State([])

    with gr.Tab('Chat'):
        with gr.Column():
            chatbot = gr.Chatbot(
                shared.convo_history,
                height=768,
                type='messages',
                show_copy_button=True,
            )
            input_textbox_str = gr.Textbox()
            clear_button = gr.Button("Clear")
        with gr.Column():
            audio1 = gr.Audio()
            state_mostrecent_reply = gr.State([])


        def request_llm_response_to_chat_history(chatbot_history_to_send_to_llm: List[dict]) -> List[dict]:
            role = None
            resp = ''
            for content, response_role in vllm_util.yield_streaming_response(chatbot_history_to_send_to_llm, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, True):
                resp += content
                role = role or response_role
                yield chatbot_history_to_send_to_llm + [{'role': role, 'content': resp}]
            return chatbot_history_to_send_to_llm

        # def generate_audio_from_latest_response(chatbot_history: List[dict]):
        #     return

        input_textbox_str.submit(
            update_history_with_user_prompt,
            [input_textbox_str, chatbot],
            [input_textbox_str, chatbot],
            scroll_to_output=True,
            queue=False,
        ) \
            .then(request_llm_response_to_chat_history, chatbot, chatbot)
            # .then(generate_audio_from_latest_response, chatbot, something)

        clear_button.click(lambda: None, None, chatbot, queue=False)
    with gr.Tab('Audio'):
        gr.Chatbot()


demo.launch(
    show_error=True,
    #         inline: bool | None = None,
    #         inbrowser: bool = False,
    #         share: bool | None = None,
    #         debug: bool = False,
    #         max_threads: int = 40,
    #         auth: (
    #             Callable[[str, str], bool] | tuple[str, str] | list[tuple[str, str]] | None
    #         ) = None,
    #         auth_message: str | None = None,
    #         prevent_thread_lock: bool = False,
    #         show_error: bool = False,
    #         server_name: str | None = None,
    #         server_port: int | None = None,
    #         *,
    #         height: int = 500,
    #         width: int | str = "100%",
    #         favicon_path: str | None = None,
    #         ssl_keyfile: str | None = None,
    #         ssl_certfile: str | None = None,
    #         ssl_keyfile_password: str | None = None,
    #         ssl_verify: bool = True,
    #         quiet: bool = False,
    #         show_api: bool = not wasm_utils.IS_WASM,
    #         allowed_paths: list[str] | None = None,
    #         blocked_paths: list[str] | None = None,
    #         root_path: str | None = None,
    #         app_kwargs: dict[str, Any] | None = None,
    #         state_session_capacity: int = 10000,
    #         share_server_address: str | None = None,
    #         share_server_protocol: Literal["http", "https"] | None = None,
    #         auth_dependency: Callable[[fastapi.Request], str | None] | None = None,
    #         max_file_size: str | int | None = None,
    #         _frontend: bool = True,
    #         enable_monitoring: bool | None = None,
)
