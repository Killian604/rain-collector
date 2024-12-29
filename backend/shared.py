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
    possible_tape_prompts: List[str] = [line.strip() for line in f.readlines() if line.strip()]
print(f'{possible_tape_prompts=}')
# possible_tape_prompts = ['This is the default text. Read me out loud for Laura.']
parler_names = ['Laura', 'Gary', 'Jon', 'Lea', 'Karen', 'Rick', 'Brenda', 'David', 'Eileen', 'Jordan', 'Mike', 'Yann',
                'Joy', 'James', 'Eric', 'Lauren', 'Rose', 'Will', 'Jason', 'Aaron', 'Naomie', 'Alisa', 'Patrick',
                'Jerry', 'Tina', 'Jenna', 'Bill', 'Tom', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Bruce', 'Emily',
                ]
# parler_names = [
#     'Gary',
# ]
possible_callers_descriptions: List[str] = [
    '$NAME is a crazy person from Utah. This person speaks English. There is lots of background noise. There is a radio hiss of static in the background as if this is recorded from a pay phone.'
]

def generate_caller_desc():
    name = random.choice(parler_names)
    return random.choice(possible_callers_descriptions).replace('$NAME', name)
