# Last updated: not working
from typing import Optional
import argparse

from backend import vllm_util
import gradio as gr



def build_demo(host, port, model: Optional[str] = None):
    with gr.Blocks(
        theme='gradio/monochrome',
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# vLLM text completion demo\n")
        inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")
        inputbox2 = gr.Textbox(label="Input", value=vllm_util.get_models(host, port)[0])

        outputbox = gr.Textbox(label="Output", placeholder="Generated result from the model", lines=5, max_lines=300)
        inputbox.submit(vllm_util.yield_streaming_response, [inputbox, inputbox2], [outputbox])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-url", type=str, default="http://localhost:8000/v1/completions")
    args = parser.parse_args()
    host, port = 'localhost', 8000
    demo = build_demo(host, port)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=False)
