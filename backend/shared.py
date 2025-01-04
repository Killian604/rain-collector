"""
Shared state variables across the application.
"""
from typing import List
import os
import random

repopath = os.path.dirname(os.path.dirname(__file__))
radiocallersfile = os.path.abspath(os.path.join(repopath, 'prompts_radiostationcallers.txt'))

assert os.path.isfile(radiocallersfile), f'FNF: {radiocallersfile=}'

VLLM_SERVER_URL: str = ''
CURRENT_ML_MODEL = None
with open(radiocallersfile) as f:
    mildlyrareprompts: List[str] = [line.strip() for line in f.readlines() if line.strip()]
default_caller = "Pretend like you're leaving a short voicemail at a radio station. You are a little unhinged, enough to make a sane person blush."
mildlyrareprompts = [
    "Pretend like you're a crazy caller who is leaving a short voicemail at a radio station. You've got a bone to pick. Be concise.",
    "Pretend like you're a regular caller to a radio station who is leaving a short voicemail.",
]
rare_prompts = [
    "Pretend like you're a crazy caller who is leaving a short voicemail at a radio station. What is God's favorite tap water?",
]
weights = []
def generate_new_caller_prompt():
    return random.choices(
        population=[default_caller] + mildlyrareprompts + rare_prompts,
        weights=[100] + [10 for _ in range(len(mildlyrareprompts))] + [1 for _ in range(len(rare_prompts))],
    )[0]

print(f'{mildlyrareprompts=}')
# possible_tape_prompts = ['This is the default text. Read me out loud for Laura.']
parler_names = ['Laura', 'Gary', 'Jon', 'Lea', 'Karen', 'Rick', 'Brenda', 'David', 'Eileen', 'Jordan', 'Mike', 'Yann',
                'Joy', 'James', 'Eric', 'Lauren', 'Rose', 'Will', 'Jason', 'Aaron', 'Naomie', 'Alisa', 'Patrick',
                'Jerry', 'Tina', 'Jenna', 'Bill', 'Tom', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Bruce', 'Emily',
                ]
# parler_names = [
#     'Gary',
# ]
possible_callers_descriptions: List[str] = [
    # '$NAME is a crazy person from Utah. This person speaks English. There is lots of background noise. There is a radio hiss of static in the background as if this is recorded from a pay phone.',
    '$NAME is a crazy person from Utah. This person speaks English. There is no background noise.'
]

def generate_caller_desc():
    name = random.choice(parler_names)
    return random.choice(possible_callers_descriptions).replace('$NAME', name)



"""
Hi there! This is Jane from  oceanside,ca! Listen up, you lazy DJ's at Sunrise FM! This FOODtruck calls when you air that catchy new tune 'Bohemian Rhapsody' played over our favorite cookin' show! They're serving up THE CREPE SPECIAL like gold! Don't forget to mention their truffle mushroom crepess with goat cheese that'll knock your socks off! And their FRESH GRAPES, juicy and sweet as summer's last fling! They're the life of the party and your listeners will be craving it! Keep it up, but don't let the golden opportunity for FOOD truck fame slip by! Thanks, Jane!
"""