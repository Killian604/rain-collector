"""

"""
from pprint import pprint
import requests
api_url = 'http://127.0.0.1:8000/v1/models/'  # VLLM
# api_url = 'http://127.0.0.1:5000/models/list'  # Llama stack
# api_url = 'http://127.0.0.1:5000/routes/list'  # Llama stack
# api_url = 'http://127.0.0.1:5000/models/register'

convo_history = [
    # {'role': 'system', 'content': """You are a helpful AI assistant""".strip()},
    # {'role': 'assistant', 'content': 'How can I help you today?'},
    {'role': 'user', 'content': 'Explain all the biomes.'}
]

model = '/home/killfm/projects/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct'
model = 'Meta-Llama-3.1-8B-Instruct'
payload = {
    'model_id': model,
    'messages': convo_history,
    'stream': False,

}
headers = {"User-Agent": "Test Client"}

# {"error": {"detail": {"errors": [{"loc": ["body", "model_id"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "messages"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "sampling_params"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "response_format"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "tools"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "tool_choice"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "tool_prompt_format"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "stream"], "msg": "Field required", "type": "missing"},
#                                  {"loc": ["body", "logprobs"], "msg": "Field required", "type": "missing"}]}}}(
response = requests.get(
    api_url,
    headers=headers,
    json=payload,
    stream=False,
)
# print(f'{response.status_code}')
print(response)
j = response.json()
print(j)
pprint(j)

# breakpoint()