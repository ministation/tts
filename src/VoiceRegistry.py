import os
import glob
import shutil

import yaml

BUILTIN_SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")
SEX_VALUES = ("Male", "Female", "Unsexed")
ENGINES = ("silero", "xtts")
SEX_FALLBACK = {
    "Male": "aidar",
    "Female": "xenia",
    "Unsexed": "baya",
}


def _voices_dir():
    return os.path.join(os.getcwd(), "voices")


def _config_path():
    return os.path.join(_voices_dir(), "config.yml")


def load_config():
    path = _config_path()
    if not os.path.isfile(path):
        return {"builtin": {}, "voices": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"builtin": {}, "voices": {}}


def save_config(config):
    os.makedirs(_voices_dir(), exist_ok=True)
    with open(_config_path(), "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def get_reference_path(speaker_id, voices_dir=None):
    voices_dir = voices_dir or _voices_dir()
    return os.path.join(voices_dir, speaker_id, "reference.wav")


def has_silero_model(speaker_id, voices_dir=None):
    voices_dir = voices_dir or _voices_dir()
    return os.path.isfile(os.path.join(voices_dir, f"{speaker_id}.pt"))


def has_clone_reference(speaker_id, voices_dir=None):
    return os.path.isfile(get_reference_path(speaker_id, voices_dir))


def has_model(speaker_id, voices_dir=None):
    entry = get_voice_config(speaker_id)
    if entry and entry.get("engine") == "xtts":
        return has_clone_reference(speaker_id, voices_dir)
    return has_silero_model(speaker_id, voices_dir) or has_clone_reference(speaker_id, voices_dir)


def get_voice_config(speaker_id, config=None):
    config = config or load_config()
    return (config.get("voices") or {}).get(speaker_id)


def get_engine(speaker_id, config=None):
    entry = get_voice_config(speaker_id, config)
    if entry:
        return entry.get("engine", "silero")
    if has_clone_reference(speaker_id):
        return "xtts"
    return "silero"


def uses_custom_model(speaker_id, config=None):
    entry = get_voice_config(speaker_id, config)
    if not entry:
        return has_silero_model(speaker_id) or has_clone_reference(speaker_id)
    if entry.get("source", "builtin") != "custom":
        return False
    engine = entry.get("engine", "silero")
    if engine == "xtts":
        return has_clone_reference(speaker_id)
    return has_silero_model(speaker_id)


def is_xtts_voice(speaker_id, config=None):
    entry = get_voice_config(speaker_id, config)
    return entry and entry.get("engine") == "xtts" and entry.get("source") == "custom" and has_clone_reference(speaker_id)


def get_fallback(speaker_id, config=None):
    config = config or load_config()
    entry = get_voice_config(speaker_id, config)
    if entry:
        fallback = entry.get("fallback")
        if fallback in BUILTIN_SPEAKERS:
            return fallback
        sex = entry.get("sex", "Unsexed")
        return SEX_FALLBACK.get(sex, "baya")
    return None


def get_voice_entry(speaker_id, config=None):
    config = config or load_config()
    if speaker_id in BUILTIN_SPEAKERS:
        meta = (config.get("builtin") or {}).get(speaker_id, {})
        return {
            "id": speaker_id,
            "name": meta.get("name", speaker_id),
            "sex": meta.get("sex", "Unsexed"),
            "description": meta.get("description", ""),
            "builtin": True,
            "has_model": True,
            "uses_custom": False,
            "source": "builtin",
            "engine": "silero",
            "active_speaker": speaker_id,
            "fallback": None,
        }
    entry = get_voice_config(speaker_id, config)
    if entry:
        custom = uses_custom_model(speaker_id, config)
        fallback = get_fallback(speaker_id, config)
        engine = entry.get("engine", "silero")
        active = "xtts-clone" if engine == "xtts" and custom else ("random" if custom else fallback)
        return {
            "id": speaker_id,
            "name": entry.get("name", speaker_id),
            "sex": entry.get("sex", "Unsexed"),
            "description": entry.get("description", ""),
            "builtin": False,
            "has_model": has_model(speaker_id),
            "uses_custom": custom,
            "source": entry.get("source", "builtin"),
            "engine": engine,
            "active_speaker": active,
            "fallback": fallback,
            "round_start": entry.get("round_start", True),
        }
    return None


def list_all_voices(model_speakers, include_random=False):
    config = load_config()
    voices_dir = _voices_dir()
    seen = set()
    result = []

    for speaker_id in BUILTIN_SPEAKERS:
        if speaker_id in model_speakers or speaker_id in (config.get("builtin") or {}):
            entry = get_voice_entry(speaker_id, config)
            if entry:
                seen.add(speaker_id)
                result.append(entry)

    for speaker_id in (config.get("voices") or {}):
        if speaker_id not in seen:
            entry = get_voice_entry(speaker_id, config)
            if entry:
                seen.add(speaker_id)
                result.append(entry)

    for pt_path in glob.glob(os.path.join(voices_dir, "*.pt")):
        speaker_id = os.path.basename(pt_path).replace(".pt", "")
        if speaker_id not in seen:
            seen.add(speaker_id)
            result.append({
                "id": speaker_id,
                "name": speaker_id,
                "sex": "Unsexed",
                "description": "Silero .pt",
                "builtin": False,
                "has_model": True,
                "uses_custom": True,
                "source": "custom",
                "engine": "silero",
                "active_speaker": "random",
                "fallback": None,
            })

    if include_random and "random" in model_speakers:
        result.append({
            "id": "random",
            "name": "Случайный",
            "sex": "Unsexed",
            "description": "Случайный Silero",
            "builtin": True,
            "has_model": True,
            "uses_custom": False,
            "source": "builtin",
            "engine": "silero",
            "active_speaker": "random",
            "fallback": None,
        })

    return sorted(result, key=lambda v: v["id"])


def list_speaker_ids(model_speakers):
    return [v["id"] for v in list_all_voices(model_speakers)]


def set_voice_source(speaker_id, source):
    if source not in ("builtin", "custom"):
        raise ValueError("source must be builtin or custom")
    config = load_config()
    voices = config.setdefault("voices", {})
    if speaker_id not in voices:
        raise KeyError(f"Voice '{speaker_id}' not found in config.yml")
    voices[speaker_id]["source"] = source
    save_config(config)


def install_clone_voice(speaker_id, reference_path, name, sex="Unsexed", fallback=None, description=""):
    profile_dir = os.path.join(_voices_dir(), speaker_id)
    os.makedirs(profile_dir, exist_ok=True)
    ref_dst = get_reference_path(speaker_id)
    shutil.copy2(reference_path, ref_dst)
    old_pt = os.path.join(_voices_dir(), f"{speaker_id}.pt")
    if os.path.isfile(old_pt):
        os.remove(old_pt)
    add_voice_to_config(speaker_id, name, sex, fallback, description, source="custom", engine="xtts")
    return {"reference": ref_dst, "profile_dir": profile_dir}


def add_voice_to_config(speaker_id, name, sex="Unsexed", fallback=None, description="", source="builtin", engine="silero"):
    if sex not in SEX_VALUES:
        raise ValueError(f"sex must be one of {SEX_VALUES}")
    if fallback is None:
        fallback = SEX_FALLBACK.get(sex, "baya")
    if fallback not in BUILTIN_SPEAKERS:
        raise ValueError(f"fallback must be one of {BUILTIN_SPEAKERS}")
    if source not in ("builtin", "custom"):
        raise ValueError("source must be builtin or custom")
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")

    config = load_config()
    voices = config.setdefault("voices", {})
    voices[speaker_id] = {
        "name": name,
        "sex": sex,
        "fallback": fallback,
        "source": source,
        "engine": engine,
        "description": description,
    }
    save_config(config)
