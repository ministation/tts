import os
import glob

from src.VoiceRegistry import get_fallback, load_config, uses_custom_model

cwd = os.getcwd()

dynVoices = cwd + "/voices"

if not os.path.exists(dynVoices):
    os.makedirs(dynVoices)

# фикс не доступных спикеров (алиасы из игр — WC3, Dota, HL и т.д.)
speakers_not_avaible = {
    'Male': ['arthas', 'thrall', 'kael', 'rexxar', 'furion', 'illidan', 'kelthuzad', 'narrator', 'cairne', 'garithos', 'anubarak', 'uther', 'grunt', 'medivh', 'villagerm', 'illidan_f', 'peon', 'chen', 'dread_bm', 'priest', 'acolyte', 'muradin', 'dread_t', 'mannoroth', 'peasant', 'wheatley', 'barney', 'raynor', 'tusk', 'earth', 'wraith', 'bristle', 'gyro', 'treant', 'lancer', 'clockwerk', 'batrider', 'kotl', 'kunkka', 'pudge', 'juggernaut', 'vort_e2', 'omni', 'sniper', 'skywrath', 'huskar', 'bloodseeker', 'shaker', 'storm', 'tide', 'riki', 'witchdoctor', 'doom', 'bandit', 'pantheon', 'tychus', 'breen', 'kleiner', 'father', 'tosh', 'stetmann', 'hanson', 'swann', 'hill', 'gman_e2', 'valerian', 'gman', 'vort', 'aradesh', 'dornan', 'harris', 'cabbot', 'decker', 'dick', 'officer', 'frank', 'gizmo', 'hakunin', 'harold', 'harry', 'maxson', 'killian', 'lieutenant', 'loxley', 'lynette', 'marcus', 'master', 'morpheus', 'overseer', 'rhombus', 'set', 'sulik', 'dude', 'archmage', 'demoman', 'engineer', 'heavy', 'medic', 'scout', 'soldier', 'spy', 'admiral', 'alchemist', 'archimonde', 'breaker', 'captain', 'footman', 'grom', 'hh', 'keeper', 'naga_m', 'naga_rg', 'rifleman', 'satyr', 'voljin', 'sidorovich', 'oleg', 'bartender', 'security'],
    'Female': ['maiev', 'tyrande', 'jaina', 'ladyvashj', 'naisha', 'sylvanas', 'sorceress', 'alyx', 'glados', 'announcer', 'kerrigan', 'lina', 'luna', 'windranger', 'templar', 'ranger', 'mortred', 'queen', 'evelynn', 'elder', 'jain', 'laura', 'nicole', 'tandi', 'vree', 'huntress', 'peasant_w', 'sylvanas_w', 'zina', 'doctor'],
    'Unsexed': ['meepo', 'bounty', 'antimage', 'yuumi', 'myron', 'dryad', 'elf_eng', 'scientist', 'clown']
}

speakers_rnd = {}


def _registry_fallback(speaker):
    config = load_config()
    fb = get_fallback(speaker, config)
    if fb:
        return fb
    return None


def SpeakerPatchInit(model, example_text):
    for idx, value in enumerate(speakers_not_avaible['Male']):
        speakers_rnd[value] = 1 if (idx % 2) == 0 else 2
    for idx, value in enumerate(speakers_not_avaible['Female']):
        speakers_rnd[value] = 1 if (idx % 2) == 0 else 2


def GetAllSpeakers(speakers):
    from src.VoiceRegistry import list_speaker_ids
    return list_speaker_ids(speakers)


def SpeakerPatch(speaker, speakers):
    voiceFile = dynVoices + "/" + speaker + ".pt"
    if uses_custom_model(speaker) and os.path.exists(voiceFile):
        return "random", voiceFile

    registry_fb = _registry_fallback(speaker)
    if registry_fb:
        return registry_fb, voiceFile

    if speaker not in speakers:
        if speaker in speakers_not_avaible['Male']:
            speaker = 'aidar' if speakers_rnd.get(speaker, 1) == 1 else 'eugene'
        elif speaker in speakers_not_avaible['Female']:
            speaker = 'xenia' if speakers_rnd.get(speaker, 1) == 1 else 'kseniya'
        elif speaker in speakers_not_avaible['Unsexed']:
            speaker = 'baya'

    if speaker not in speakers:
        speaker = 'baya'
    return speaker, voiceFile
