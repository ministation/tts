import base64, os, io, sys

import re
from stressrnn import StressRNN

from src.SpeakerPatch import SpeakerPatch, SpeakerPatchInit, GetAllSpeakers
from src.VoiceRegistry import (
    list_all_voices,
    has_model,
    add_voice_to_config,
    set_voice_source,
    BUILTIN_SPEAKERS,
    is_xtts_voice,
    is_piper_voice,
    get_reference_paths,
    get_voice_model_path,
    get_piper_model_path,
)
from src.TrainingJobs import create_job, get_job, run_job
from src.AudioPrep import prepare_references, prepare_training_uploads, MAX_REFERENCE_FILES, MAX_TRAIN_FILES
from src.VoiceCloner import train_voice_from_upload
from src.CloneClient import health as clone_health, is_available as clone_available
from src.WarmUp import WarmUp
from src.SoundEffects import add_echo, add_radio_effect, add_robot

import torch

import soundfile as sf

# ffmpeg fix
current_directory = os.path.abspath(os.path.dirname(__file__))
bin_directory = os.path.join(current_directory, 'bin')
os.environ['PATH'] = f"{bin_directory}:{os.environ['PATH']}"
sys.path.append(bin_directory)

accent = StressRNN()

#print(torchaudio.list_audio_backends())

torch.set_num_threads(int(os.environ.get("threads","6")))
#torchaudio.set_audio_backend("sox")

deviceName = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(deviceName)

ApiToken = os.environ.get("apitoken","test")

local_file = 'model.pt'
if not os.path.isfile(local_file):
    print("Start download silero models")
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
example_text = "В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров."

model.to(device)  # gpu or cpu

from flask import Flask, request, jsonify, abort, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

app = Flask(__name__, static_folder='www')


@app.errorhandler(HTTPException)
def handle_http_exception(exc):
    return jsonify({"description": exc.description, "error": exc.name}), exc.code

#import logging
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
#app.logger.disabled = True
#log.disabled = True


# доступные спикера
speakers = model.speakers # ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']

SpeakerPatchInit(model,example_text)

@app.route('/')
def index():
    return send_from_directory('www', 'index.html')


@app.route('/train')
def train_page():
    return send_from_directory('www', 'train.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('www', path)

@app.route('/voices')
def api_voices():
    if request.args.get('detailed') == '1':
        return jsonify(list_all_voices(speakers))
    return jsonify(GetAllSpeakers(speakers))


@app.route('/voices/upload', methods=['POST'])
def api_upload_voice():
    req = request.json
    if not req or req.get('api_token') != ApiToken:
        abort(403)
    speaker = req.get('speaker')
    if not speaker or not re.match(r'^[a-z][a-z0-9_]{0,31}$', speaker):
        abort(400, description="Invalid speaker id (lowercase latin, digits, underscore)")
    if speaker in ('random',):
        abort(400, description="Cannot overwrite reserved speaker")
    data_b64 = req.get('file')
    if not data_b64:
        abort(400, description="Missing file (base64-encoded .pt)")
    try:
        data = base64.b64decode(data_b64)
    except Exception:
        abort(400, description="Invalid base64 in file field")
    if len(data) < 100:
        abort(400, description="File too small to be a valid voice model")
    voice_path = os.path.join('voices', f'{speaker}.pt')
    with open(voice_path, 'wb') as f:
        f.write(data)
    if req.get('register'):
        add_voice_to_config(
            speaker,
            req.get('name', speaker),
            req.get('sex', 'Unsexed'),
            req.get('fallback'),
            req.get('description', ''),
            source='custom',
        )
    else:
        try:
            set_voice_source(speaker, 'custom')
        except KeyError:
            pass
    return jsonify({
        'ok': True,
        'speaker': speaker,
        'has_model': has_model(speaker),
        'path': voice_path,
    })


@app.route('/voices/train', methods=['POST'])
def api_train_voice():
    if request.form.get('api_token') != ApiToken:
        abort(403, description="Неверный API токен")
    speaker = (request.form.get('speaker') or '').strip().lower()
    if not speaker or not re.match(r'^[a-z][a-z0-9_]{0,31}$', speaker):
        abort(400, description="ID голоса: только латиница, цифры и _, начинается с буквы (например: ivan)")
    if speaker in BUILTIN_SPEAKERS or speaker == 'random':
        abort(400, description=f"ID «{speaker}» зарезервирован, выберите другое имя")

    name = request.form.get('name') or speaker
    sex = request.form.get('sex', 'Unsexed')
    if sex not in ('Male', 'Female', 'Unsexed'):
        abort(400, description="Invalid sex")
    fallback = request.form.get('fallback')
    engine = (request.form.get('engine') or 'piper').strip().lower()
    if engine not in ('piper', 'silero', 'xtts'):
        abort(400, description="engine должен быть piper, silero или xtts")
    description = request.form.get('description') or {
        'piper': 'Обучен Piper (ONNX, CPU)',
        'xtts': 'Клонирован по образцу (XTTS)',
        'silero': 'Обучен по образцу (Silero)',
    }.get(engine, 'Кастомный голос')

    audio_files = []
    for key in ('audio', 'audio[]'):
        audio_files.extend(request.files.getlist(key))
    audio_files = [f for f in audio_files if f and f.filename]
    if not audio_files:
        abort(400, description="Не выбраны аудиофайлы. Для Piper лучше 5–10+ минут речи.")
    max_files = MAX_TRAIN_FILES if engine == 'piper' else MAX_REFERENCE_FILES
    if len(audio_files) > max_files:
        abort(400, description=f"Слишком много файлов (максимум {max_files})")

    if engine == 'xtts' and not clone_available():
        abort(503, description="XTTS не установлен. Запустите: powershell -File scripts/setup_clone.ps1")
    if engine == 'piper':
        from src.PiperEngine import is_piper_available
        from src.PiperTrainer import is_whisper_available
        if not is_whisper_available():
            abort(
                503,
                description="Для Piper нужен faster-whisper. Запустите: powershell -File scripts/setup_piper.ps1",
            )

    import shutil
    import tempfile

    refs_dir = tempfile.mkdtemp(prefix="voice_upload_")
    prepared = []
    uploads = []
    try:
        for audio in audio_files:
            raw = tempfile.NamedTemporaryFile(delete=False, suffix='.upload')
            audio.save(raw.name)
            raw.close()
            if os.path.getsize(raw.name) < 1024:
                raise ValueError(f"Файл {audio.filename} слишком маленький или пустой")
            uploads.append(raw.name)
        if engine == 'piper':
            prepared = prepare_training_uploads(uploads, refs_dir)
        else:
            prepared = prepare_references(uploads, refs_dir)
    except Exception as exc:
        shutil.rmtree(refs_dir, ignore_errors=True)
        abort(400, description=f"Ошибка обработки аудио: {exc}")
    finally:
        for path in uploads:
            if os.path.isfile(path):
                os.remove(path)

    job_id = create_job()

    def work(progress):
        try:
            return train_voice_from_upload(
                model,
                speaker,
                prepared,
                name,
                sex,
                fallback,
                description,
                engine=engine,
                progress=progress,
            )
        finally:
            shutil.rmtree(refs_dir, ignore_errors=True)

    run_job(job_id, work)
    return jsonify({'job_id': job_id, 'engine': engine, 'references': len(prepared)})


@app.route('/voices/train/status', methods=['GET'])
def api_train_stack_status():
    from src.PiperTrainer import piper_status
    from src.PiperEngine import is_piper_available

    status = piper_status()
    status['piper_runtime'] = is_piper_available()
    status['xtts_installed'] = clone_available()
    status['xtts_running'] = clone_health()
    return jsonify(status)


@app.route('/voices/clone/status', methods=['GET'])
def api_clone_status():
    # legacy endpoint — теперь отражает Piper stack
    from src.PiperTrainer import piper_status
    from src.PiperEngine import is_piper_available

    st = piper_status()
    return jsonify({
        'installed': is_piper_available() or st.get('whisper'),
        'running': True,
        'engine': 'piper',
        'details': st,
        'xtts_installed': clone_available(),
        'xtts_running': clone_health(),
    })


@app.route('/voices/train/<job_id>', methods=['GET'])
def api_train_status(job_id):
    job = get_job(job_id)
    if not job:
        abort(404, description="Job not found")
    return jsonify(job)

# Docker HealthCheck
@app.route('/health', methods=['GET'])
def doHEALTH():
    return "OK"

# TTS вход
@app.route('/tts', methods=['POST'])
def doTTS():
    req = request.json
    if 'api_token' not in req:
        abort(400, description="Missing api_token")
        return
    if req['api_token'] != ApiToken:
        abort(403)
        return
    if 'text' not in req:
        abort(400, description="Missing text")
        return
    if 'speaker' not in req:
        abort(400, description="Missing speaker")
        return
    audio = None
    speaker = req['speaker']

    if 'sample_rate' not in req:
        req['sample_rate'] = 24000
    if 'put_accent' not in req:
        req['put_accent'] = True
    if 'put_yo' not in req:
        req['put_yo'] = False
    if 'format' not in req:
        req['format'] = 'ogg'

    if is_piper_voice(req['speaker']):
        from src.PiperEngine import synthesize as piper_synthesize
        # Piper не понимает маркеры ударения Silero (+) и читает их вслух
        text = plain_text_for_piper(req['text'], put_yo=req['put_yo'])
        audio, piper_sr = piper_synthesize(text, speaker_id=req['speaker'])
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if piper_sr != req['sample_rate']:
            duration = len(audio) / float(piper_sr)
            new_len = max(1, int(duration * req['sample_rate']))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
    elif is_xtts_voice(req['speaker']):
        voice_model = get_voice_model_path(req['speaker'])
        refs = get_reference_paths(req['speaker'])
        from src.CloneClient import synthesize as clone_synthesize, encode_voice
        text = plain_text_for_piper(req['text'])
        if not os.path.isfile(voice_model):
            if not refs:
                abort(404, description=f"Voice model not found: {req['speaker']}.pt")
            encode_voice(refs, voice_model, speaker_id=req['speaker'])
        clone_result = clone_synthesize(
            text,
            voice_model_path=voice_model,
            speaker_id=req['speaker'],
            reference_paths=refs or None,
        )
        wav_bytes = base64.b64decode(clone_result['audio'])
        audio, clone_sr = sf.read(io.BytesIO(wav_bytes))
        if clone_sr != req['sample_rate']:
            import numpy as np
            duration = len(audio) / clone_sr
            new_len = int(duration * req['sample_rate'])
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )
    else:
        speaker, voiceFile = SpeakerPatch(req['speaker'], speakers)

        if 'ssml' in req and req['ssml']:
            audio = model.apply_tts(ssml_text=patch_ssml(req['text']),
                speaker=speaker,
                sample_rate=req['sample_rate'],
                put_accent=req['put_accent'],
                put_yo=req['put_yo'],
                voice_path=voiceFile
            )
        else:
            audio = model.apply_tts(text=patch_text(req['text'], put_yo=req['put_yo']),
                speaker=speaker,
                sample_rate=req['sample_rate'],
                put_accent=req['put_accent'],
                put_yo=req['put_yo'],
                voice_path=voiceFile
            )
    # Saving to bytes buffer — всегда mono (N,), иначе (1,N) пишется как N каналов
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[-1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=-1)
    audio = np.ascontiguousarray(audio.reshape(-1))

    with io.BytesIO() as buffer_:
        if req["format"] == "ogg":
            sf.write(buffer_, audio, req['sample_rate'], format="ogg", subtype="VORBIS")
        else:
            sf.write(buffer_, audio, req['sample_rate'], format=req["format"])
        #torchaudio.save(buffer_, audio.unsqueeze(0), req['sample_rate'], format=req["format"])
        buffer_.seek(0)

        effect = None
        if 'effect' in req:
            effect = req['effect']

        if effect == "Echo": 
            buffer_ = add_echo(buffer_, output_format=req["format"])
        elif effect == "Radio":
            buffer_ = add_radio_effect(buffer_, req['sample_rate'], format=req["format"])
        elif effect == "Robot":
            buffer_ = add_robot(buffer_, req['sample_rate'], format=req["format"])

        return jsonify({'results': [{'audio': base64.b64encode(buffer_.getvalue()).decode()}]})

def patch_ssml(ssml_content):
    def add_accents(match):
        text = match.group(1)
        try:
            accented_text = accent.put_stress(text)
        except:
            accented_text = text  # Если не удалось поставить ударение, возвращаем исходный текст
        return f">{accented_text}<"

    # Заменяем текст внутри тегов
    patched_ssml = re.sub(r'>([^<>]+)<', add_accents, ssml_content)
    return patched_ssml

def patch_text(text_content, put_yo=False):
    text = ""
    try:
        text = accent.put_stress(text_content)
    except:
        text = text_content  # Если не удалось поставить ударение, возвращаем исходный текст
    if not put_yo:
        text = _keep_original_yo(text_content, text)
    return text


def _keep_original_yo(original, processed):
    """StressRNN часто меняет «е»→«ё»; оставляем «ё» только если оно было в исходнике."""
    if not processed:
        return processed
    orig_letters = [c for c in str(original) if c.isalpha()]
    out = []
    i = 0
    for ch in processed:
        if ch == "+":
            out.append(ch)
            continue
        if ch in ("ё", "Ё") and i < len(orig_letters):
            src = orig_letters[i]
            if src in ("е", "Е"):
                out.append("е" if ch == "ё" else "Е")
            else:
                out.append(ch)
            i += 1
        else:
            if ch.isalpha():
                i += 1
            out.append(ch)
    return "".join(out)


def plain_text_for_piper(text_content, put_yo=False):
    """Текст для Piper: только буквы/цифры; одиночные ъ/ь не оставляем (читаются как названия)."""
    if not text_content:
        return ""
    text = str(text_content)
    # markup / скобки целиком
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    # всё, что не буква и не цифра → пробел
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    # одиночные ъ/ь → espeak говорит «твёрдый/мягкий знак»
    text = re.sub(r"(?<![а-яА-ЯёЁa-zA-Z])[ъьЪЬ]+(?![а-яА-ЯёЁa-zA-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == '__main__':
    WarmUp(model,speakers)
    app.run(host= '0.0.0.0',debug=False,port=int(os.environ.get("PORT","8000")))
