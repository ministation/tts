import os
import shutil
import tempfile

import numpy as np
import torch

from src.VoiceTrainer import PITCH_RANGES, _gender_penalty, _warmup_random

SAMPLE_RATE = 24000
EMBED_SR = 16000
TEST_PHRASE = "Привет, это тестовая фраза для обучения голоса на станции."
DEFAULT_CANDIDATES = 60
_encoder = None


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def _voice_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.getcwd(), "models", "spkrec-ecapa"),
            run_opts={"device": "cpu"},
        )
    return _encoder


def _as_numpy(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _to_tensor(audio, sr):
    import librosa

    y = _as_numpy(audio)
    if sr != EMBED_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=EMBED_SR)
    return torch.from_numpy(y).unsqueeze(0)


def _embed_audio(audio, sr=EMBED_SR):
    encoder = _voice_encoder()
    with torch.no_grad():
        embed = encoder.encode_batch(_to_tensor(audio, sr))
    return embed.squeeze().cpu().numpy()


def _embed_file(path):
    import librosa

    audio, sr = librosa.load(path, sr=EMBED_SR, mono=True)
    return _embed_audio(audio, sr)


def _embed_reference(paths):
    embeds = [_embed_file(path) for path in paths]
    if len(embeds) == 1:
        return embeds[0]
    return np.mean(embeds, axis=0)


def _estimate_pitch(audio, sr=SAMPLE_RATE):
    import librosa

    audio = _as_numpy(audio)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=60, fmax=400)
    values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            values.append(p)
    if not values:
        return 0.0
    return float(np.median(values))


def train_from_references(model, reference_paths, sex="Unsexed", candidates=DEFAULT_CANDIDATES, progress=None):
    if isinstance(reference_paths, str):
        reference_paths = [reference_paths]
    reference_paths = [p for p in reference_paths if p and os.path.isfile(p)]
    if not reference_paths:
        raise ValueError("Нет reference-файлов для обучения")

    if progress:
        progress(progress=5, message="Загрузка модели сравнения голосов...")
    ref_embed = _embed_reference(reference_paths)

    import librosa

    ref_audio, _ = librosa.load(reference_paths[0], sr=SAMPLE_RATE, mono=True)
    ref_pitch = _estimate_pitch(ref_audio, SAMPLE_RATE)

    tmp_dir = tempfile.mkdtemp(prefix="silero_train_")
    best_score = -1.0
    best_path = None
    best_pitch = 0.0
    best_similarity = 0.0

    for idx in range(1, candidates + 1):
        if progress:
            progress(
                progress=int(5 + (idx - 1) / candidates * 85),
                total=candidates,
                message=f"Подбор Silero-модели {idx}/{candidates}...",
            )

        candidate_pt = os.path.join(tmp_dir, f"candidate_{idx:03d}.pt")
        _warmup_random(model)
        model.save_random_voice(candidate_pt)

        sample_audio = model.apply_tts(
            text=TEST_PHRASE,
            speaker="random",
            voice_path=candidate_pt,
            sample_rate=SAMPLE_RATE,
            put_accent=True,
            put_yo=False,
        )
        cand_embed = _embed_audio(sample_audio, SAMPLE_RATE)
        cand_pitch = _estimate_pitch(sample_audio, SAMPLE_RATE)

        similarity = _cosine(ref_embed, cand_embed)
        pitch_delta = abs(ref_pitch - cand_pitch) / (ref_pitch + 1.0)
        penalty = _gender_penalty(cand_pitch, sex)
        score = similarity - pitch_delta * 0.08 - penalty

        if score > best_score:
            best_score = score
            best_path = candidate_pt
            best_pitch = cand_pitch
            best_similarity = similarity

    if not best_path:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Не удалось подобрать Silero-модель")

    return {
        "score": round(best_score, 4),
        "similarity": round(best_similarity, 4),
        "pitch_hz": round(best_pitch, 1),
        "reference_pitch_hz": round(ref_pitch, 1),
        "model_path": best_path,
        "candidates_tested": candidates,
        "tmp_dir": tmp_dir,
    }


def install_trained_model(speaker_id, trained_model_path, voice_model_path, tmp_dir=None):
    os.makedirs(os.path.dirname(voice_model_path) or ".", exist_ok=True)
    shutil.copy2(trained_model_path, voice_model_path)
    if tmp_dir and os.path.basename(tmp_dir).startswith("silero_train_"):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return voice_model_path
