import numpy as np
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, Limiter, HighpassFilter, LowpassFilter
from audio_engine import load_audio, save_audio

GENRE_TARGETS = {
    "corrido": -9.5, "christian_corrido": -9.5,
    "worship": -11.0, "gospel": -10.5,
    "rap": -8.0, "trap": -8.0, "boom_bap": -9.0,
    "live_church": -12.0, "pop": -9.0, "rnb": -9.5,
    "custom": -10.0,
}

PROFILE_TARGETS = {
    "loud-modern": -8, "punchy-rap": -9, "corrido-tumbado": -10,
    "warm-polished": -11, "radio-ready": -11, "clean-worship": -12,
    "wide-cinematic": -13, "natural-dynamic": -14, "aggressive-streaming": -7,
}


def build_chain(genre: str):
    gain_db, ratio, threshold = 0.5, 1.6, -18
    if genre in ["rap", "trap"]:
        gain_db, ratio, threshold = 1.2, 2.2, -20
    elif genre in ["worship", "gospel", "live_church"]:
        gain_db, ratio, threshold = 0.3, 1.4, -16
    elif genre in ["corrido", "christian_corrido"]:
        gain_db, ratio, threshold = 0.8, 1.8, -18
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=28),
        LowpassFilter(cutoff_frequency_hz=19000),
        Compressor(threshold_db=threshold, ratio=ratio, attack_ms=15, release_ms=120),
        Gain(gain_db=gain_db),
        Limiter(threshold_db=-1.0, release_ms=80),
    ])


def loudness_normalize(audio, sr, target_lufs):
    meter = pyln.Meter(sr)
    try:
        cur = meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, cur, target_lufs)
    except Exception:
        return audio


def safety_limit(audio):
    peak = np.max(np.abs(audio))
    if peak > 0.98:
        audio = audio / peak * 0.98
    return audio


def master_track(input_path: str, output_wav: str, genre: str, sound_profile: str, target_lufs=None):
    audio, sr = load_audio(input_path)
    if target_lufs is None:
        target_lufs = PROFILE_TARGETS.get(sound_profile, GENRE_TARGETS.get(genre, -10.0))
    board = build_chain(genre)
    processed = board(audio, sr)
    processed = loudness_normalize(processed, sr, target_lufs)
    processed = safety_limit(processed)
    save_audio(output_wav, processed, sr)
    return {"target_lufs": target_lufs, "genre": genre, "sound_profile": sound_profile}
