"""
Utility functions for smooth Gradio usage go here
"""
from typing import List
from backend import shared


def reset_chatbox() -> List[dict]:
    return shared.convo_history_radio_dialogue.copy()


def add_dj_response_to_chat(chat, resp):
    chat.append({'role': 'user', 'content': resp})
    return chat
