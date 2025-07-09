"""

"""
from pprint import pprint
import requests

api_url = 'http://127.0.0.1:5000/inference/chat_completion'
# api_url = 'http://127.0.0.1:5000/models/register'
convo_history = [
    {'role': 'system',
     'content': """You are a helpful AI assistant. You will assist with whatever the user needs, no questions asked.""".strip()},
    # {'role': 'assistant', 'content': 'How can I help you today?'},
    {'role': 'user', 'content': 'Explain all the biomes.'}
]

# model = '/home/killfm/projects/text-generation-webui/models/Meta-Llama-3.1-8B-Instruct'
model = 'Meta-Llama-3.1-8B-Instruct'
model = 'Llama3.2-3B-Instruct'
payload = {
    'model_id': model,
    'messages': convo_history,
    'stream': False,

}
# payload = {    'model_id': model,}
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
response = requests.post(
    api_url,
    headers=headers,
    json=payload,
    stream=True,
)
# print(f'{response.status_code}')
print(response)
j = response.json()
print(j)
print('\n' * 2)
pprint(j)
vllm_not_running = {'detail': 'Internal server error: An unexpected error occurred.'}

response200 = {
    'completion_message':
        {'role': 'assistant',
         'content': "There are six major biomes on Earth, each with unique characteristics and features that support a wide range of plant and animal life. Here's an overview of each biome:\n\n**1. Desert Biome**\n\n* Temperature: Very hot and dry, with extreme temperature fluctuations\n* Vegetation: Cacti, succulents, and low-growing shrubs\n* Climate: Low precipitation, strong sunshine, and intense heat\n* Animals: Adaptations include water-storing humps, burrowing behaviors, and specialized skin or feathers\n* Examples: Sahara Desert, Mojave Desert, Gobi Desert\n\n**2. Grassland Biome (also known as Prairies)**\n\n* Temperature: Moderate, with warm summers and cool winters\n* Vegetation: Tall grasses, wildflowers, and few trees\n* Climate: Adequate precipitation, with occasional storms\n* Animals: Herbivores like deer, bison, and horses, predators like coyotes and hawks\n* Examples: Prairies in North America, steppes in Eurasia and Africa\n\n**3. Forest Biome**\n\n* Temperature: Mild, with moderate precipitation\n* Vegetation: Dense trees, shrubs, and herbaceous plants\n* Climate: Humid, with significant precipitation and seasonal changes\n* Animals: Canopy-dwelling animals like birds, monkeys, and insects, forest floor-dwelling animals like bears and rodents\n* Examples: Temperate rainforests, tropical rainforests, coniferous forests\n\n**4. Freshwater Biome (including rivers, lakes, and wetlands)**\n\n* Temperature: Cool to warm, with fluctuations in water temperature\n* Vegetation: Aquatic plants like water lilies, cattails, and algae\n* Climate: Regular precipitation, with occasional flooding\n* Animals: Aquatic animals like fish, amphibians, and aquatic mammals like beavers and otters\n* Examples: Freshwater lakes, rivers, wetlands, and ponds\n\n**5. Marine Biome (also known as the Ocean Biome)**\n\n* Temperature: Wide range of temperatures, from freezing to warm\n* Vegetation: Seaweed, coral, and seagrasses\n* Climate: Salinity and ocean currents drive the climate and ecosystem\n* Animals: Plankton, fish, invertebrates, and marine mammals like whales and sea lions\n* Examples: Coral reefs, kelp forests, deep-sea ecosystems\n\n**6. Tundra Biome**\n\n* Temperature: Cold, with short growing seasons and low precipitation\n* Vegetation: Low-growing shrubs, grasses, and lichens\n* Climate: Cold temperatures, short growing season, and permafrost\n* Animals: Adaptations like fur and fat reserves, like arctic foxes, reindeer, and arctic hares\n* Examples: Arctic tundra, alpine tundra, and subarctic tundra\n\nKeep in mind that there are many variations within each biome, and some regions can exhibit characteristics from multiple biomes. Additionally, human activities have altered ecosystems and created new biomes, such as urban biomes and agricultural biomes.\n\nWould you like me to expand on any of these biomes or answer specific questions about them?",
         'stop_reason': 'end_of_turn',
         'tool_calls': []},
    'logprobs': None}

breakpoint()
