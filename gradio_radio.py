"""
For running radio app

Usage: `gradio <ThisFileName>`
"""
from typing import Dict, List, Optional, Tuple
from backend import gradio_logic, logging_extra as log, shared, vllm_util
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
# import chromadb
import gradio as gr
import os
import random
import re
import soundfile as sf
import time
import torch
# from TTS.api import TTS

# Magic variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Define the vLLM server URL
VLLM_SERVER_IP, VLLM_SERVER_PORT = 'localhost', 8000
VLLM_SERVER_IP, VLLM_SERVER_PORT = '10.0.0.73', 8000
voice_clone_wav_fp = '/home/killfm/Videos/aj/aj_cryptic_andjustthatlittlebit.wav'
reference_voice_text = "And just that little bit makes me wild. And so, they can get them, because they're powerful, they're smart, they're neat"
DEFAULT_TEMP = 1.0
MAX_TOKENS = 512

if gr.NO_RELOAD:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", ).to(device)
    # print(f'\n{TTS().list_models()=}\n')
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    torch.compile()


assert os.path.isfile(voice_clone_wav_fp)


# Save shared
shared.VLLM_SERVER_URL = vllm_util.create_vllm_chat_uri(VLLM_SERVER_IP, VLLM_SERVER_PORT)  # f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}/v1/chat/completions"  # Update with your actual server URL
shared.CURRENT_ML_MODEL = vllm_util.get_models(VLLM_SERVER_IP, VLLM_SERVER_PORT)[0]

# Init default convo
# TODO: Fill context history
convo_history = [
    {'role': 'system', 'content': """
    You are an AI that generates realistic dialogue in English only. You consider the request and then generate an appropriate response that follows the request.
    Although some speakers may be dramatic, do not reply in all-caps.
""".strip()},
    {'role': 'assistant', 'content': 'What type of text should I generate for you today?'},
]


# Pure functions that help with Gradio processing
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
    """Given a chat history, prompt LLM to return response."""
    role = None
    print(f'{chatbot_history_to_send_to_llm=}')
    resp = ''
    for content, response_role in vllm_util.yield_streaming_response(chatbot_history_to_send_to_llm, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, max_tokens=MAX_TOKENS, stream=True):
        resp += content
        role = role or response_role
        yield chatbot_history_to_send_to_llm + [{'role': role, 'content': resp}]
    return chatbot_history_to_send_to_llm


@torch.inference_mode()
def infer_to_file_parler(tts_text: str, output_fp: str, description_of_speaker: str, **kwargs):
    # PARLER SECTION
    # pad_token_id = model.config.pad_token_id
    description_ids = tokenizer(description_of_speaker, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(tts_text, return_tensors="pt").input_ids.to(device)
    generation_kwargs = {
        "attention_mask": torch.ones_like(description_ids),
        'temperature': 0.7,

    }
    with log.Timer('Logging Parler response generation timing'):
        generation = model.generate(
            input_ids=description_ids,
            prompt_input_ids=prompt_ids,
            **generation_kwargs
        )
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_fp, audio_arr, model.config.sampling_rate)

@torch.inference_mode()
def infer_to_file_coqui(tts_text: str, output_fp, reference_audio_fp: Optional[str] = None, **kwargs):
    # COQUI SECTION

    # Run TTS
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    # wav = tts.tts(
    #     text=reference_voice_text,
    #     language="en",
    #     speaker_wav=voiceclonefile,
    #     emotion=None,
    #     speed=1.0,
    # )
    # # print(f'{wav=} // {type(wav)=}')
    print(f'{tts_text=}')
    # Text to speech to a file
    _output_fp: str = tts.tts_to_file(
        text=tts_text,
        language="en",
        speaker_wav=reference_audio_fp,
        emotion='energetic',
        speed=1.5,
        file_path=output_fp,
        split_sentences=True,
    )
    # print(f'tts.tts_to_file returns: {type(_output_fp)=} // {_output_fp=}')


def click_generate_tape(prompt: Optional[str] = None, output_fp: Optional[str] = None, debug: bool = True):
    """

    :param prompt:
    :param output_fp:
    :return:
    """
    print(f'Fetching recording...')
    prompt: str = prompt or random.choice(shared.possible_tape_prompts)
    prompt: str = random.choice(shared.possible_tape_prompts)  # TODO: Remove me later
    if debug:
        print(f'{prompt=} / {type(prompt)=}')
    output_fp = output_fp or os.path.join(shared.repopath, 'output_audio', time.strftime('%y%m%d-%H%M%S') + '.mp3')

    # prompt = "Hey, how are you doing today?"
    convo_history_a = convo_history.copy()
    convo_history_a.append({'role': 'user', 'content': prompt})
    if debug:
        print(f'{convo_history_a=}')
    with log.Timer('Logging VLLM response time'):
        content, _role = vllm_util.generate_response(convo_history_a, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, False, max_tokens=MAX_TOKENS, temperature=DEFAULT_TEMP)
    # parler_text = ""
    # for content, _response_role in vllm_util.yield_streaming_response(convo_history_a, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, max_tokens=MAX_TOKENS, temperature=DEFAULT_TEMP, stream=True):
    #     parler_text += content
    tts_text = re.sub(r'\s+', ' ', content.strip())
    print(f'\n\n{tts_text=}\n\n')
    # prompt = """
    # A delight surrounding the Mathematics of Finance is that while much is known, so
    # much is unknown. Consequently, with the current state of understanding, it is wise
    # to...destroy all humans. Kick her in the pussy.
    # """
    # description = "A young female speaker delivers a slightly expressive and animated speech with a medium slow speed and pitch. The recording sounds like a clip from a radio station."
    # description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    description = shared.generate_caller_desc()
    # parler_text = description

    # infer_to_file_coqui(tts_text, voice_clone_wav_fp, output_fp)
    infer_to_file_parler(tts_text, output_fp, description)
    print(f'Output written to: {output_fp}')
    return output_fp


# GRADIO BLOCKS
with gr.Blocks(
        theme='gradio/monochrome',
        analytics_enabled=False,
        title='YAPP Radio!',
        fill_height=True,
        fill_width=True,
) as demo:
    # with gr.Tab('Audio'):
    with gr.Row():
        state_mostrecent_reply = gr.State([None])
        btn_new_recording = gr.Button('Listen to new voice recording')
        audio_msg = gr.Audio(
            label='Listen to tape',
            sources=['upload'],
            type='filepath',
        )
        btn_new_recording.click(click_generate_tape, None, audio_msg)

    with gr.Row():
        pass
    # with gr.Tab('Chat'):
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(convo_history, height=768, type='messages', show_copy_button=True, )
            input_textbox_str = gr.Textbox()
            clear_button = gr.Button("Clear")
        with gr.Column():
            audio1 = gr.Audio()


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



demo.launch(
    show_error=True,
    debug=True,
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
