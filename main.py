"""Stem mixing: balance levels, light EQ per stem, sum to a stereo mix."""
from __future__ import annotations

from typing import Dict

import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, HighShelfFilter, LowShelfFilter

from audio_engine import load_audio, write_wav, normalize_loudness, true_peak_limit


# Rough per-stem gain targets (dB) relative to unity.
STEM_GAINS_DB: Dict[str, float] = {
    "vocals": 0.0,
    "vocal": 0.0,
    "drums": -2.0,
    "bass": -3.0,
    "instrumental": -3.0,
    "beat": -3.0,
    "music": -3.0,
    "other": -4.0,
}


def _gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return audio * (10 ** (gain_db / 20.0))


def _process_stem(audio: np.ndarray, sr: int, kind: str) -> np.ndarray:
    kind = (kind or "other").lower()
    if "voc" in kind:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=90.0),
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.0, q=0.7),
            Compressor(threshold_db=-18.0, ratio=3.0, attack_ms=5.0, release_ms=80.0),
        ])
    elif "bass" in kind:
        board = Pedalboard([
            LowShelfFilter(cutoff_frequency_hz=80.0, gain_db=1.5, q=0.7),
            Compressor(threshold_db=-16.0, ratio=2.5, attack_ms=10.0, release_ms=120.0),
        ])
    elif "drum" in kind or "beat" in kind:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=35.0),
            Compressor(threshold_db=-14.0, ratio=2.0, attack_ms=8.0, release_ms=100.0),
        ])
    else:
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=40.0),
            Compressor(threshold_db=-16.0, ratio=2.0, attack_ms=10.0, release_ms=120.0),
        ])
    out = board(audio, sr)
    gain = STEM_GAINS_DB.get(kind, -4.0)
    return _gain(np.asarray(out, dtype=np.float32), gain)


def mix_stems(stems: Dict[str, str], output_path: str) -> str:
    """`stems` is a dict mapping stem kind ("vocals", "drums", ...) -> file path."""
    if not stems:
        raise ValueError("No stems provided")

    rendered = []
    sr_ref = None
    max_len = 0
    for kind, path in stems.items():
        audio, sr = load_audio(path)
        if sr_ref is None:
            sr_ref = sr
        rendered.append((_process_stem(audio, sr, kind), sr))
        max_len = max(max_len, audio.shape[0])

    # Pad/sum to common length, mono->stereo if needed.
    summed = np.zeros((max_len, 2), dtype=np.float32)
    for audio, _ in rendered:
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        if audio.shape[0] < max_len:
            pad = np.zeros((max_len - audio.shape[0], audio.shape[1]), dtype=np.float32)
            audio = np.vstack([audio, pad])
        summed += audio[:max_len, :2]

    summed = normalize_loudness(summed, sr_ref or 44100, target_lufs=-14.0)
    summed = true_peak_limit(summed, ceiling_db=-1.5)
    write_wav(output_path, summed, sr_ref or 44100)
    return output_path
