"""
Shared state variables across the application.
"""
from typing import List
import os
import random

repopath = os.path.dirname(os.path.dirname(__file__))
VLLM_SERVER_URL: str = ''
CURRENT_ML_MODEL = None
sample_rate_kokoro = 24_000

station_metadata = """
- Current radio station info: 102 point 7 "Yap Radio"
- Current radio show live: Mystery Caller Hour. Only interesting callers allowed!
- Current location: Rain City, Washington
- Current weather: 2 degree Celsius, raining heavily
- Caller criteria: any callers from the continental United States. No callers from Rain City.
"""

convo_history_general = [
    {'role': 'system', 'content': 'You are a general-purpose AI. You help the user. Answer concisely.'},
    # {'role': 'user', 'content': 'Please generate a random location in the United States of America.'},
]
convo_history_character_generator = [
    {'role': 'system', 'content': f"""
You are a character-generating AI for a radio station. You make up radio station callers and 
their associated dialogue. You will be given a character and topic, and then your job is to generate
 a corresponding response that follows the request.
The following is a list that should be avoided:
- do not reply in all-caps and do not include quotations around your response
- do not include the tone or loudness of the caller
- do not include any non-spoken parts of language (e.g. do not include "(giggles)" or "(In a panicked tone)" or "*sigh*")
- do not include any metadata. Only reply with the dialogue of the character
- No run-on sentences (each sentence must be 400 characters or less)
- only reply in English. No other languages.

Current station metadata:
{station_metadata}""".strip()},
    {'role': 'assistant', 'content': 'What character dialogue should I generate? Include the details below.'}, ]

convo_history_radio_dialogue = [
    {'role': 'system', 'content': f"""
Below is a conversation between a caller and radio DJ. You are the caller, and the user is the radio DJ.
The caller is usually a little weird, and you should follow the topic and personality of
the caller. Note, you should always reply in English, and you should refrain from all-caps unless absolutely
necessary. Keep the response concise, topical, and to the point.

Current station metadata:
{station_metadata}
""".strip().replace('\n', ' ')}, ]

chathistory_yield_location = [
    {'role': 'system', 'content': 'You are a general-purpose AI. You help the user. Answer concisely.'},
    {'role': 'user', 'content': 'Please generate a random location in the United States of America.'},
]
# radiocallersfile = os.path.abspath(os.path.join(repopath, 'prompts_radiostationcallers.txt'))
# assert os.path.isfile(radiocallersfile), f'FNF: {radiocallersfile=}'
# with open(radiocallersfile) as f:
#     mildlyrareprompts: List[str] = [line.strip() for line in f.readlines() if line.strip()]

default_callers = [
    "Pretend like you're leaving a short voicemail at a radio station. ",
]
mildlyrareprompts = [
    "Pretend like you're a crazy caller who is leaving a short voicemail at a radio station. You've got a bone to pick. Be concise.",
    "Pretend like you're a crazy caller to a radio station who is leaving a short voicemail.",
]
rare_prompts = [
    "Pretend like you're a crazy caller who is leaving a short voicemail at a radio station. What is God's favorite tap water?",
]


def generate_new_caller_prompt():
    return random.choices(
        population=default_callers + mildlyrareprompts + rare_prompts,
        weights=[10 / len(default_callers) for _ in default_callers]
                + [10 / len(mildlyrareprompts) for _ in range(len(mildlyrareprompts))]
                + [1 for _ in range(len(rare_prompts))],
    )[0]


# possible_tape_prompts = ['This is the default text. Read me out loud for Laura.']
parler_names = ['Laura', 'Gary', 'Jon', 'Lea', 'Karen', 'Rick', 'Brenda', 'David', 'Eileen', 'Jordan', 'Mike', 'Yann',
                'Joy', 'James', 'Eric', 'Lauren', 'Rose', 'Will', 'Jason', 'Aaron', 'Naomie', 'Alisa', 'Patrick',
                'Jerry', 'Tina', 'Jenna', 'Bill', 'Tom', 'Carol', 'Barbara', 'Rebecca', 'Anna', 'Bruce', 'Emily',
                ]
possible_callers_descriptions: List[str] = [
    # '$NAME is a crazy person from Utah. This person speaks English. There is lots of background noise. There is a radio hiss of static in the background as if this is recorded from a pay phone.',
    '$NAME is a crazy person from Utah. This person speaks English. There is no background noise.'
]



def generate_caller_desc():  # For Parler TTS
    name = random.choice(parler_names)
    return random.choice(possible_callers_descriptions).replace('$NAME', name)


xtts_speakers_female = [
    'Claribel Dervla',  # Robotic, throaty album store owner
    'Daisy Studious',  # Girly squirrel voice
    'Gracie Wise',  # A future Tammy
    'Tammie Ema',  # Strong, self-confident
    'Alison Dietlinde',  # British, small
    'Ana Florence',  # Close to Rachel Evan Woods maybe
    'Annmarie Nele',  # British, _
    'Asya Anara',  # Older British
    'Brenda Stern',  # Southern Belle?
    'Gitta Nikolina',  # Hidden Ireland
    'Henriette Usha',  # British, _
    'Sofia Hellen',  # British, _
    'Tammy Grit',
    'Tanja Adelina',  # Standard American. Good
    'Vjollca Johnnie', 'Andrew Chipper', 'Badr Odhiambo', 'Dionisio Schuyler',
    'Royston Min', 'Viktor Eka', 'Abrahan Mack', 'Adde Michal', 'Baldur Sanjin', 'Craig Gutsy', 'Damien Black',
    'Gilberto Mathias', 'Ilkin Urbano',
    'Kazuhiko Atallah',  # Male, mystery accent. Queen's English
    'Ludvig Milivoj', 'Suad Qasim', 'Torcull Diarmuid',
    'Viktor Menelaos', 'Zacharie Aimilios', 'Nova Hogarth', 'Maja Ruoho', 'Uta Obando', 'Lidiya Szekeres',
    'Chandra MacFarland', 'Szofi Granger', 'Camilla Holmström', 'Lilya Stainthorpe', 'Zofija Kendrick', 'Narelle Moon',
    'Barbora MacLean', 'Alexandra Hisakawa', 'Alma María', 'Rosemary Okafor',
]

xtts_speakers_male = [
    'Ige Behringer',  # Rural Brit
    'Filip Traverse',  # Brit
    'Damjan Chapman',  # Some form of Queen's English
    'Wulf Carlevaro',  # Middle Eatern Brit
    'Aaron Dreschner',  # Hidden posh Brit
    'Kumar Dahl',  # 10/10 Deep resonant, strong
    'Eugenio Mataracı',  # Regular. Librarian probably.
    'Ferran Simen',  # Mostly British?
    'Xavier Hayasaka',  # British
    'Luis Moray',  # Regular?
    'Marcos Rudaski',  # British
]


random_female_names = [
    'Anna',
    'Ava',
    'Brenda',
    'Lily',
    'Emily',
    'Sophia',
    'Tammy',
    'Charlotte',
    'Carrie',
]

random_male_names = [
    'Ethan', 'Mason', 'Noah', 'Lucas', 'Charles', 'Oliver', 'Marco',
]

random_locations = [
    'USA',
]


all_male_firstnames = random_male_names + [x.split(' ')[0] for x in xtts_speakers_male]


def generate_random_xtts_caller():
    """
    TODO: evaluate if we should even keep this
    :return:
    """
    is_male = bool(random.randint(0, 1))
    name = random.choice(all_male_firstnames) if is_male else random.choice(random_female_names)
    location = random.choice(random_locations)

"""Hi there! This is Jane from  oceanside,ca! Listen up, you lazy DJ's at Sunrise FM! This FOODtruck calls when you air that catchy new tune 'Bohemian Rhapsody' played over our favorite cookin' show! They're serving up THE CREPE SPECIAL like gold! Don't forget to mention their truffle mushroom crepess with goat cheese that'll knock your socks off! And their FRESH GRAPES, juicy and sweet as summer's last fling! They're the life of the party and your listeners will be craving it! Keep it up, but don't let the golden opportunity for FOOD truck fame slip by! Thanks, Jane!"""
"""Hi, this is Emily Wilson calling in to the Mystery Caller Hour. I'm planning a double date with my boyfriend and his brother next weekend, and I'm really nervous about meeting his brother who's 6'5" and has a prosthetic leg. I'm wondering if I should wear heels or flats. Got any advice?"""
