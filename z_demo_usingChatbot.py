import gradio as gr

def chat_greeter(msg: str, history: list[dict]) -> list[dict]:
    history.append({"role": "assistant", "content": "Hello!"})
    return history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(chat_greeter, [msg, chatbot], [chatbot])

demo.launch()
