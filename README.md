# rain-collector

Collect your memories and customize your very own personalized AI. Emphasis on privacy.

# Installation

1. Clone the repo

More instructions TBD



# Downloading Llama 3.1
1. Visit host site: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
2. Request access to the model
3. Generate a Huggingface token
4. `huggingface-cli download --repo-type model --token $YOURHFTOKEN meta-llama/Meta-Llama-3.1-8B`
    - Minor note: the default download location is: `~/.cache/huggingface/hub/`