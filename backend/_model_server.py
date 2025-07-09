"""
Implementation for serving LLM responses
"""
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from enum import Enum
from typing import List, Optional, Literal, Union, Iterator, Dict
import re
import datetime
import os
import json
# import jwt
# import llama_types as llama_cpp
# from llm_utils import client
# from retrivial_ranking import search_context, search_context_with_time
# from settings import *


from fastapi import FastAPI, Depends
import requests

fastapiapp = FastAPI(
    title="Memory Server",
    version="0.0.1",
)

fastapiapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptResponse(BaseModel):
    output_text: str


# Routes

fastapiapp.post(
    '/'
)

fastapiapp.get(
    '/'
)


def get_llm_response(prompt: str) -> dict :
    response = requests.post(
        "http://localhost:8001/generate",
        json={"prompt": prompt},
    )
    return response.json()


@fastapiapp.get('/')
async def home() -> str:
    return ''


@fastapiapp.get("/predict")
async def predict(prompt: str):
    return get_llm_response(prompt)


@fastapiapp.get("/predaudio")
async def predaudio(prompt: str):
    return get_llm_response(prompt)


