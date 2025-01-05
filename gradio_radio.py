"""
For running radio app

Usage: `gradio <ThisFileName>`
"""
# import rccoqui
# from rccoqui.TTS import tts

# from rccoqui.TTS.tts.configs.xtts_config import XttsConfig
# from rccoqui.TTS.tts.models.xtts import Xtts
# from rccoqui.TTS.api import TTS
from typing import List, Optional, Tuple
from backend import logging_extra as log, shared, vllm_util
from tempfile import TemporaryDirectory
import gradio as gr
import numpy as np
import os
import random
import re
import soundfile as sf
from transformers import pipeline
from TTS.api import TTS
import time
import torch

# import chromadb
# from parler_tts import ParlerTTSForConditionalGeneration
# from pydub.audio_segment import read_wav_audio
from backend.gradio_util import reset_chatbox, add_dj_response_to_chat

tab_title = '102.7FM YAPP Radio!'
IS_LIVE = False  # If True, autoplay audio


# Magic variables
# Define the vLLM server URL
VLLM_SERVER_IP, VLLM_SERVER_PORT = 'localhost', 8000
VLLM_SERVER_IP, VLLM_SERVER_PORT = '10.0.0.73', 8000
# voice_clone_wav_fp = '/home/killfm/Videos/aj/aj_cryptic_andjustthatlittlebit.wav'  # aj_evan
voice_clone_wav_fp = '/home/killfm/Videos/aj/aj_evangelion.wav'
wav_ref_audio_files = [
    '/home/killfm/Videos/aj/aj_evangelion.wav',
    '/home/killfm/Videos/voices/dana.mp3',
]
reference_voice_text = "And just that little bit makes me wild. And so, they can get them, because they're powerful, they're smart, they're neat"
device = 'cuda'  # "cuda:0" if torch.cuda.is_available() else "cpu"
default_speech_speed = 1.0
DEFAULT_TEMP = 0.8
MAX_TOKENS = 2048
debug = True
js_head = """
<script>

</script>
"""

if gr.NO_RELOAD:
    # ['Claribel Dervla', 'Daisy Studious', 'Gracie Wise', 'Tammie Ema', 'Alison Dietlinde', 'Ana Florence', 'Annmarie Nele', 'Asya Anara', 'Brenda Stern', 'Gitta Nikolina', 'Henriette Usha', 'Sofia Hellen', 'Tammy Grit', 'Tanja Adelina', 'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler', 'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black', 'Gilberto Mathias', 'Ilkin Urbano', 'Kazuhiko Atallah', 'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid', 'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres', 'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon', 'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor', 'Ige Behringer', 'Filip Traverse', 'Damjan Chapman', 'Wulf Carlevaro', 'Aaron Dreschner', 'Kumar Dahl', 'Eugenio Mataracı', 'Ferran Simen', 'Xavier Hayasaka', 'Luis Moray', 'Marcos Rudaski']
    # configlocation = '/home/killfm/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json'
    # modellocation = '/home/killfm/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2'
    # config = XttsConfig()
    # config.load_json(configlocation)
    # model2 = Xtts.init_from_config(config)
    # model2.load_checkpoint(config, checkpoint_path=modellocation, eval=True)
    # model2.cuda()
    # torch.compile(model2)

    model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)
    model.eval()
    # model = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
    # model = TTS("tts_models/multilingual/multi-dataset/bark").to(device)  # Limited for voice cloning, but interesting to say the least
    # print(f'\n{TTS().list_models()=}\n')
    # model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    # tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    torch.compile(model)
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device_map='auto')

    # outputs: dict = model2.synthesize(
    #     "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    #     config,
    #     speaker_wav="/data/TTS-public/_refclips/3.wav",
    #     gpt_cond_len=3,
    #     language="en",
    # )
    # waveform = outputs['wav']


assert os.path.isfile(voice_clone_wav_fp), f'Voice clone file not found: {voice_clone_wav_fp=}'
assert all([os.path.isfile(x) for x in wav_ref_audio_files]), f'fnf: {wav_ref_audio_files}'

# Save shared
shared.VLLM_SERVER_URL = vllm_util.create_vllm_chat_uri(VLLM_SERVER_IP, VLLM_SERVER_PORT)  # f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}/v1/chat/completions"  # Update with your actual server URL
shared.CURRENT_ML_MODEL = vllm_util.get_models(VLLM_SERVER_IP, VLLM_SERVER_PORT)[0]

# Init default convo



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
def infer_to_file_coqui(tts_text: str, output_fp, reference_audio_fp: Optional[str] = None, speed=default_speech_speed, **kwargs):
    # Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # tts_text = '[laughter] Is this the bark model? Are you kidding me? 99.5 FM? I cant believe my luck!!!!'
    # print(f'{tts_text=}')
    # Text to speech to a file

    print(f'{reference_audio_fp=}')

    with log.Timer('COQUI'), TemporaryDirectory() as td:
        _output_fp: str = model.tts_to_file(
            text=tts_text,
            language="en",
            # speaker_wav=reference_audio_fp,
            speed=speed,
            file_path=output_fp,
            split_sentences=True,
            speaker='Tammie Ema',

        )
        # y, sr = librosa.load(tempoutpath)
        # print(f'{y, sr=}')
        # print(f'{len(y)=}')
        # cleaned_y = librosa.effects.split(y, top_db=40, hop_length=int(0.5*sr))
        # print(f'{len(cleaned_y)=}')
        # write_wav(output_fp.replace('mp3', 'wav'), sr, cleaned_y)
        # # librosa.output.write_wav(output_fp, cleaned_y, sr)

    # print(f'tts.tts_to_file returns: {type(_output_fp)=} // {_output_fp=}')
    return output_fp

def generate_new_caller_tape() -> str:
    """

    :param prompt:
    :param output_fp:
    :return:
    """
    print(f'Fetching recording...')
    # prompt: str = random.choice(shared.mildlyrareprompts)
    prompt: str = shared.generate_new_caller_prompt()
    if debug:
        print(f'{prompt=} / {type(prompt)=}')

    convo_history_a = shared.convo_history_character_generator.copy()
    convo_history_a.append({'role': 'user', 'content': prompt})

    # if debug:
    #     print(f'{convo_history_a=}')

    with log.Timer('Logging VLLM response time'):
        content, _role = vllm_util.generate_response(convo_history_a, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, stream=False, max_tokens=MAX_TOKENS, temperature=DEFAULT_TEMP)
    tts_text = re.sub(r'\s+', ' ', content.strip()).replace('\n', ' ')

    # print(f'\n\n{tts_text=}\n\n')
    # prompt = """
    # A delight surrounding the Mathematics of Finance is that while much is known, so
    # much is unknown. Consequently, with the current state of understanding, it is wise
    # to...destroy all humans. Kick her in the pussy.
    # """
    # description = "A young female speaker delivers a slightly expressive and animated speech with a medium slow speed and pitch. The recording sounds like a clip from a radio station."
    # description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    description = shared.generate_caller_desc()
    if tts_text.startswith('('):
        return generate_new_caller_tape()
    return tts_text

def generate_reply(chat) -> str:
    with log.Timer('Logging VLLM response time'):
        content, _role = vllm_util.generate_response(chat, shared.CURRENT_ML_MODEL, shared.VLLM_SERVER_URL, stream=False, max_tokens=MAX_TOKENS, temperature=DEFAULT_TEMP)
    tts_text = re.sub(r'\s+', ' ', content.strip()).replace('\n', ' ')
    return tts_text


def render_voice(tts_text: str, speed: float, callerwavpath):
    # TODO: pull random voice?
    print(f'{callerwavpath=}')
    output_fp_mp3 = os.path.join(shared.repopath, 'output_audio', time.strftime('%y%m%d-%H%M%S') + '.mp3')
    output_text_fp = output_fp_mp3.replace('.mp3', '.txt')
    with open(output_text_fp, 'w') as f:
        f.write(tts_text + '\n')
    infer_to_file_coqui(tts_text, output_fp_mp3, reference_audio_fp=callerwavpath, speed=speed)
    # infer_to_file_parler(tts_text, output_fp, description)
    if debug:
        print(f'Output written to: {output_fp_mp3}')
    return output_fp_mp3


def add_caller_text_to_chatbox(chat, text):
    chat.append({
        'role': 'assistant',
        'content': text,
    })
    return chat

def add_most_recent_chatbox_reply_to_current_most_recent_state(chatbox):
    return chatbox[-1]['content']

def transcribe(audio: tuple):
    """

    :param audio: 2-tuple[sample rate, numpyarray np.int16].
    :return: Returns transcription
    """
    # print(f'{audio=}')
    if not audio:
        return ''
    start = time.perf_counter()
    sr, y = audio
    y = y.astype(np.float32)
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y /= np.max(np.abs(y))  # Dev note: it divides by it's own loudest value rather than max int16?
    end = time.perf_counter()
    print(f'Time to infer: {round(end - start, 1)} secs')
    return transcriber({"sampling_rate": sr, "raw": y})["text"]


def set_new_caller():
    reference_wav = random.choice(wav_ref_audio_files)
    return reference_wav

def init_page():
    return


# GRADIO BLOCKS
with gr.Blocks(
        theme=gr.themes.Monochrome(),
        analytics_enabled=False,
        title=tab_title,
        # fill_height=True,
        # fill_width=True,
        head=js_head,
) as demo:
    with gr.Row():  # Top row: settings
        with gr.Column():
            slider_voicespeed = gr.Slider(label='Voice speed', value=default_speech_speed, minimum=0.1, maximum=3.0, step=0.1, visible=True, interactive=True, elem_id='slidervoicespeed')
        with gr.Column():
            pass

    with gr.Row():  # Row button: generate new caller audio
        btn_generate_new_recording = gr.Button('Receive new caller')

    with gr.Row(): pass  # Middle break

    with gr.Row():
        with gr.Column():  # Chat options and chatbox
            checkbox_show_chat = gr.Checkbox(label="Show Chat", value=True)
            select_voicesource = gr.Radio(choices=['speaker', 'wav'], label='Select source', value='speaker', interactive=True)
            chatbot = gr.Chatbot(convo_history_radio_dialogue.copy(), height=512, type='messages', show_copy_button=True, visible=checkbox_show_chat.value)
            input_textbox_str = gr.Textbox()
            clear_button = gr.Button("Disconnect Caller")
            pass
        with gr.Column():  # Audio players
            waveform_options_caller = gr.WaveformOptions(waveform_color=None, waveform_progress_color='green', trim_region_color=None, show_recording_waveform=False, sample_rate=24_000)
            waveform_options_dj = gr.WaveformOptions(waveform_color=None, waveform_progress_color='green', trim_region_color='red', show_recording_waveform=False, sample_rate=44_100)
            audio_caller = gr.Audio(label='Caller', type='filepath', show_share_button=False, show_download_button=False, autoplay=IS_LIVE, waveform_options=waveform_options_caller)
            audio_dj = gr.Audio(label='DJ', type='numpy', sources=['microphone'], show_share_button=False, waveform_options=waveform_options_dj)

    state_most_recent_tts = gr.State('')
    state_mostrecent_dj_reply = gr.State('')
    state_current_caller_wav = gr.State(random.choice(wav_ref_audio_files))

    # Functions
    # Reset chat button
    clear_button.click(reset_chatbox, None, chatbot, queue=False).then(lambda: None, None, audio_caller)

    # Click "New caller" button -> generate new caller
    btn_generate_new_recording.click(set_new_caller, inputs=None, outputs=state_current_caller_wav) \
        .then(reset_chatbox, None, chatbot) \
        .then(generate_new_caller_tape, None, state_most_recent_tts) \
        .then(add_caller_text_to_chatbox, inputs=[chatbot, state_most_recent_tts], outputs=chatbot) \
        .then(render_voice, inputs=[state_most_recent_tts, slider_voicespeed, state_current_caller_wav], outputs=audio_caller)

    # Input text into textbox: Force new message via text
    input_textbox_str.submit(update_history_with_user_prompt, [input_textbox_str, chatbot], [input_textbox_str, chatbot], scroll_to_output=True, queue=False) \
        .then(generate_reply, chatbot, state_most_recent_tts) \
        .then(add_most_recent_chatbox_reply_to_current_most_recent_state, chatbot, state_most_recent_tts) \
        .then(render_voice, inputs=[state_most_recent_tts, slider_voicespeed, state_current_caller_wav], outputs=audio_caller)

    # Pause audio recording: Create audio response from DJ
    audio_dj.stop_recording(transcribe, inputs=audio_dj, outputs=state_mostrecent_dj_reply) \
        .then(add_dj_response_to_chat, inputs=[chatbot, state_mostrecent_dj_reply], outputs=chatbot) \
        .then(generate_reply, chatbot, state_most_recent_tts) \
        .then(add_caller_text_to_chatbox, inputs=[chatbot, state_most_recent_tts], outputs=chatbot) \
        .then(render_voice, inputs=[state_most_recent_tts, slider_voicespeed, state_current_caller_wav], outputs=audio_caller)
    checkbox_show_chat.change(lambda x: gr.update(visible=x), checkbox_show_chat, chatbot)

    demo.load(init_page, inputs=None, outputs=None)

if __name__ == '__main__':
    demo.launch(
        show_error=True,
        debug=True,
        inbrowser=True,
        #         inline: bool | None = None,
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

@torch.inference_mode()
def infer_to_file_parler(tts_text: str, output_fp: str, description_of_speaker: str, **kwargs):
    # pad_token_id = model.config.pad_token_id
    # description_ids = tokenizer(description_of_speaker, return_tensors="pt").input_ids.to(device)
    # prompt_ids = tokenizer(tts_text, return_tensors="pt").input_ids.to(device)
    # generation_kwargs = {"attention_mask": torch.ones_like(description_ids), 'temperature': 0.7, }
    # with log.Timer('Logging Parler response generation timing'):
    #     generation = model.generate(
    #         prompt_input_ids=prompt_ids,
    #         input_ids=description_ids,
    #         **generation_kwargs
    #     )
    # audio_arr = generation.cpu().numpy().squeeze()
    # print(f'{audio_arr.max()=} // {audio_arr.dtype=}')
    # sf.write(output_fp, audio_arr, model.config.sampling_rate)
    # pad_token_id = model.config.pad_token_id

    input_ids = tokenizer(description_of_speaker, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(tts_text, return_tensors="pt").input_ids.to(device)
    generation_kwargs = {
        "attention_mask": torch.ones_like(input_ids),
        'temperature': 1.0,
    }

    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        **generation_kwargs
    )
    audio_arr = generation.detach().cpu().numpy().squeeze()
    sf.write(output_fp, audio_arr, model.config.sampling_rate)

