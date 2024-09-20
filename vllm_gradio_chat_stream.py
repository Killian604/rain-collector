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

# from file_monitor import WatchdogThread, UpdateThread



# if __name__ == "__main__":
#     # watchdog_thread = WatchdogThread(CHAT_PATH, NOTE_PATH)
#     # watchdog_thread.start()
#     # update_thread = UpdateThread(server_state)
#     # update_thread.start()
#
#     LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
#     LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
#     uvicorn.run(
#         app,
#         host=os.getenv("HOST", "localhost"),
#         port=os.getenv("PORT", 5000),
#     )


# Define the vLLM server URL
VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"  # Update with your actual server URL

# Function to send user message to vLLM server and get a response
def chat_with_vllm(user_message, history):
    # Update conversation history with the user's message
    history.append({"role": "user", "content": user_message})

    # Create the payload for the vLLM server request
    payload = {
        "prompt": history,
        "max_tokens": 100,  # Customize the response length as needed
    }

    # Send request to vLLM server
    response = requests.post(VLLM_SERVER_URL, json=payload)

    # Extract the model's response
    if response.status_code == 200:
        vllm_response = response.json()["generated_text"]
        # Add model response to the conversation history
        history.append({"role": "assistant", "content": vllm_response})
        return history, history  # Return the updated history to display in the chat
    else:
        return history, f"Error: {response.status_code}"


# if __name__ == '__main__':
# watchdog_thread = WatchdogThread('.')
# watchdog_thread.start()
# update_thread = UpdateThread()
# update_thread.start()
# convo_history: List[Tuple[str, str]] = [('', 'Hi, I am an AI assistant. How can I help you?')]
convo_history = [
    {'role': 'system', 'content': 'You are an AI assistant..'},
    # {'role': 'system', 'content': f'The following are the contents of "vllm_util.py":\n\n{contents}'},
    # {'role': 'assistant', 'content': 'How can I help you today?'},
    # {'role': 'user', 'content': prompt},
]
with gr.Blocks(
    theme='gradio/monochrome',
    analytics_enabled=False,
) as demo:
    chatbot = gr.Chatbot(
        convo_history,
        type='messages',
    )
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    currentmodel = vllm_util.get_models('localhost', 8000)[0]


    def user(user_message, history: List[dict]):
        """

        :param user_message:
        :param history: LIST[TUPLE[MESSAGE, RESPONSE]]

        :return:
        """
        print(f'{history=}')
        return "", history + [{'role': 'user', 'content': user_message}]  # [[user_message, None]]


    def bot(chatbot_history: List[dict]):

        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        # history[-1][1] = ""
        print(f'bot(): {chatbot_history=}')
        # chatbot_history_cleaned = [x for x in chatbot_history]
        role = None
        resp = ''
        for content, response_role in vllm_util.yield_streaming_response(chatbot_history, currentmodel, f"http://localhost:8000/v1/chat/completions/", True):
            # history[-1]['content'] += item
            resp += content
            role = role or response_role
            yield chatbot_history+[{'role': role, 'content': resp}]

        # for character in bot_message:
        #     history[-1][1] += character
        #     time.sleep(0.05)
        #     yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False) \
        .then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()


    # # Gradio UI setup
    # with gr.Blocks() as demo:
    #     chatbot = gr.Chatbot()  # Create the chatbot UI component
    #     msg = gr.Textbox(placeholder="Enter your message...")  # Input textbox for user message
    #     clear = gr.Button("Clear")  # Clear button to reset conversation
    #
    #     def respond(user_message, history):
    #         # Call the chat function to handle the conversation
    #         updated_history, chat_log = chat_with_vllm(user_message, history)
    #         return chat_log, updated_history
    #
    #     # Define the app layout and interaction
    #     msg.submit(respond, [msg, chatbot], [chatbot, chatbot])
    #     clear.click(lambda: None, None, chatbot)  # Clear chat history on button click
    #
    # # Launch the Gradio app
    # demo.launch()
