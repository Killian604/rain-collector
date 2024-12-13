"""
Demo to show how a gradio chat works
"""
import gradio as gr

# Assuming you have a function to get the LLM response
def get_prompt_response(prompt):
    # Replace this with actual API call or remote LLM invocation
    return f"Response to: {prompt}"

# Function to handle conversation tracking
def chatbot(user_message, chat_history):
    if chat_history is None:
        chat_history = []

    # Append user message to chat history
    chat_history.append(("user", user_message))

    # Get response from the LLM
    response = get_prompt_response(user_message)

    # Append LLM response to chat history
    chat_history.append(("assistant", response))

    return chat_history, chat_history

# Create Gradio interface
if __name__ == '__main__':
    with gr.Blocks(
            theme='gradio/monochrome',
            analytics_enabled=False,
    ) as demo:
        chatbot_ui = gr.Chatbot(
            layout='bubble',  #'bubble'
        )
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")

        # Conversation state for history
        state = gr.State([])

        # Chatbot UI interaction
        msg.submit(chatbot, [msg, state], [chatbot_ui, state])
        clear.click(lambda: None, None, chatbot_ui)  # Clear chat history

    # Launch the interface
    demo.launch()
