import time
import gradio as gr
from transformers import pipeline
import numpy as np

if gr.NO_RELOAD:
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe(audio: tuple):
    """

    :param audio: (tuple (sample rate, numpyarray np.int16
    :return:
    """
    if audio is None:
        return ''
    start = time.perf_counter()
    sr, y = audio
    print(f'{y.shape=} // {y.dtype=}')
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))  # Dev note: it divides by it's own loudest value rather than max int16?
    end = time.perf_counter()
    print(f'Time to infer: {round(end - start, 1)} secs')
    return transcriber({"sampling_rate": sr, "raw": y})["text"]


with gr.Blocks() as demo:
    audio_input = gr.Audio(sources=["microphone"])
    text_output = gr.Textbox(label="Transcription")

    audio_input.change(fn=transcribe, inputs=[audio_input], outputs=text_output)

demo.launch(inbrowser=True, server_port=7861)

if __name__ == '__main__':
    demo.launch(inbrowser=True, server_port=7861)
