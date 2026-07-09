import os
import shutil
import tempfile

import numpy as np

SAMPLE_RATE = 24000
TEST_PHRASE = "Привет, это тестовая фраза для обучения голоса на станции."

PITCH_RANGES = {
    "Male": (70, 175),
    "Female": (155, 280),
    "Unsexed": (80, 260),
}


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def extract_features(audio, sr=SAMPLE_RATE):
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    pitch = _estimate_pitch(y, sr)
    return np.concatenate(
        [mfcc_mean, mfcc_std, [float(cent), float(rolloff), float(zcr), float(pitch)]]
    )


def _estimate_pitch(y, sr):
    import librosa

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=60, fmax=400)
    values = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        p = pitches[idx, t]
        if p > 0:
            values.append(p)
    if not values:
        return 0.0
    return float(np.median(values))


def _gender_penalty(pitch, sex):
    low, high = PITCH_RANGES.get(sex, PITCH_RANGES["Unsexed"])
    if pitch <= 0:
        return 0.15
    if pitch < low:
        return min(0.5, (low - pitch) / low)
    if pitch > high:
        return min(0.5, (pitch - high) / high)
    return 0.0


def _synthesize_sample(model, voice_path, text=TEST_PHRASE):
    return model.apply_tts(
        text=text,
        speaker="random",
        voice_path=voice_path,
        sample_rate=SAMPLE_RATE,
        put_accent=True,
        put_yo=False,
    )


def _warmup_random(model):
    model.apply_tts(
        text="Раз, два, три.",
        speaker="random",
        sample_rate=SAMPLE_RATE,
        put_accent=True,
        put_yo=False,
    )


def train_from_reference(model, reference_path, sex="Unsexed", candidates=25, progress=None):
    import librosa

    ref_audio, _ = librosa.load(reference_path, sr=SAMPLE_RATE, mono=True)
    ref_features = extract_features(ref_audio, SAMPLE_RATE)
    ref_pitch = float(ref_features[-1])

    tmp_dir = tempfile.mkdtemp(prefix="voice_train_")
    best_score = -1.0
    best_path = None
    best_pitch = 0.0

    for idx in range(1, candidates + 1):
        if progress:
            progress(
                progress=int((idx - 1) / candidates * 100),
                total=candidates,
                message=f"Подбор кандидата {idx}/{candidates}...",
            )

        candidate_pt = os.path.join(tmp_dir, f"candidate_{idx:03d}.pt")
        _warmup_random(model)
        model.save_random_voice(candidate_pt)

        sample_audio = _synthesize_sample(model, candidate_pt)
        cand_features = extract_features(sample_audio, SAMPLE_RATE)
        cand_pitch = float(cand_features[-1])

        similarity = _cosine(ref_features, cand_features)
        pitch_delta = abs(ref_pitch - cand_pitch) / (ref_pitch + 1.0)
        penalty = _gender_penalty(cand_pitch, sex)
        score = similarity - pitch_delta * 0.15 - penalty

        if score > best_score:
            best_score = score
            best_path = candidate_pt
            best_pitch = cand_pitch

    if not best_path:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Не удалось подобрать голос")

    return {
        "score": round(best_score, 4),
        "pitch_hz": round(best_pitch, 1),
        "reference_pitch_hz": round(ref_pitch, 1),
        "model_path": best_path,
        "candidates_tested": candidates,
    }


def install_trained_voice(speaker, trained_model_path, reference_path, voices_dir=None):
    voices_dir = voices_dir or os.path.join(os.getcwd(), "voices")
    profile_dir = os.path.join(voices_dir, speaker)
    os.makedirs(profile_dir, exist_ok=True)

    model_dst = os.path.join(voices_dir, f"{speaker}.pt")
    ref_dst = os.path.join(profile_dir, "reference.wav")

    shutil.copy2(trained_model_path, model_dst)
    shutil.copy2(reference_path, ref_dst)
    tmp_parent = os.path.dirname(trained_model_path)
    if os.path.basename(tmp_parent).startswith("voice_train_"):
        shutil.rmtree(tmp_parent, ignore_errors=True)
    return {"model": model_dst, "reference": ref_dst}
