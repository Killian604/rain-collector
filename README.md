# rain-collector

Collect your memories and customize your very own personalized AI. Emphasis on privacy.

# Installation

1. Clone the repo

More instructions TBD

# Usage

`gradio vllm_gradio_chat_stream.py --watch-dirs ./backend/`
`gradio `

# Folders

frontend/ is where the JS frontend goes
backend/ is where the Python backend goes. Model is implemented and served from here.
cookbook/ is for short scripts that show you how to do stuff

# Downloading Llama 3.1
1. Visit host site: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. Request access to the model
3. Generate a Huggingface token
4. Download Llama 3.1:
  - `huggingface-cli download --repo-type model --token $HFTOKEN meta-llama/Meta-Llama-3.1-8B-Instruct`, or
    - Minor note: the default download location is: `~/.cache/huggingface/hub/`
  - `huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --token $HFTOKEN --local-dir ./models/Meta-Llama-3.1-8B-Instruct`
- `huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/*" --local-dir Meta-Llama-3.1-8B-Instruct --token $HFTOKEN`

# Ollama
- `ollama pull llama3.1`
- `ollama run llama3.1:8b-instruct-fp16`
```sh
    curl http://localhost:11434/api/chat -d '{
      "model": "llama3.1",
      "messages": [
        {
          "role": "user",
          "content": "who wrote the book godfather?"
        }
      ],
      "stream": false
    }'

```


# TTS
- `import nltk;nltk.download('averaged_perceptron_tagger_eng')`


# Notes

- Loading a GGUF model requires GGUF to be installed: `pip install gguf`


# Voice
- `fish-speech`: easy voice cloning
  - https://speech.fish.audio/inference/#http-api-inference
  - https://github.com/fishaudio/fish-speech
- `MeloTTS` has decent default voices, but fish is better for fast cloning. No ability to tune defaults.
- `tacotron2` TTS isn't as strong as other competitors
- `parler` sounds good, but bails out of the script during long blocks of text
  - https://huggingface.co/parler-tts/parler-tts-large-v1
- `Coqui/XTTS`: Not worth pursuing since development has been discontinued. 
  - It's shutting down: https://x.com/_josh_meyer_/status/1742522906041635166
  - Note: Models are saved to ~/.local/share/tts/...
- `StyleTTS2`: Incredibly hard to implement fast inference to prove concept
- `GPT-SoVITS`: Extremely hacky repo. Not enough bonuses to use this over fish-audio, so dropping usage. Written in Chinese first.
- `suno/bark`: Slow to inference. No voice cloning. Has SOME non-talking attributes like [laughter]



- To ASR a 4.5h podcast with Whisper Large, it takes about 30 min
  - So it takes about 1 min of runtime per 9 min of audio


# FAQ

- Q: How do I opt out of ChromaDB's opt-out telemetry?
- A: 
    ```python
    from chromadb.config import Settings
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # or if using PersistentClient
    client = chromadb.PersistentClient(path="/path/to/save/to", settings=Settings(anonymized_telemetry=False))

- Q: I'm missing `flash_attn`, how do I install?
- A: `pip install flash-attn --no-build-isolation`

_


https://github.com/meta-llama/llama-recipes/tree/main


# Wikipedia

## Get 
- https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz
- `gunzip enwiki-latest-all-titles-in-ns0.gz`

e

 ---

## Questions to answer:
- Where does Ollama save models after `pull`?

---

$HUGGINGFACE_HUB_CACHE
gradio vllm_gradio_chat_stream.py --watch-dirs ./backend/


# Evaluations and MMLU leaderboard
https://github.com/huggingface/evaluation-guidebook
https://github.com/huggingface/lighteval/wiki/Use-VLLM-as-backend
Source: https://github.com/huggingface/blog/blob/main/open-llm-leaderboard-mmlu.md

-Original MMLU: compare probabilities of possible answers, use highest prob between options as response
-HELM implementation: expectation is that correct answer will be highest prob, otherwise false
-AI Harness: comparison of long-form response to long-form answer
  - Self note: probs are summed (and normalized probably). If unsure or value is 
  - Docs note: "For numerical stability we gather them by summing the logarithm of the probabilities and we can decide (or not) to compute a normalization in which we divide the sum by the number of tokens to avoid giving too much advantage to longer answers "

# Other resources
https://github.com/langchain-ai/rag-from-scratch?tab=readme-ov-file
