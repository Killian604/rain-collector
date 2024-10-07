# rain-collector

Collect your memories and customize your very own personalized AI. Emphasis on privacy.

# Installation

1. Clone the repo

More instructions TBD


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

- https://speech.fish.audio/inference/#http-api-inference
- https://github.com/fishaudio/fish-speech
  - Good for voice cloning
- `MeloTTS` has decent defaults but fish is better for faster cloning
- `tacotron2` TTS isn't as strong as other competitors
- `parler` sounds good, but bails out of the script during long blocks of text
  - https://huggingface.co/parler-tts/parler-tts-large-v1
- 

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


 ---

## Questions to answer:
- Where does Ollama save models after `pull`?

---

$HUGGINGFACE_HUB_CACHE
