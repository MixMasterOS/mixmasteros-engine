"""Shared audio I/O + utility helpers used by mixing and mastering."""
from __future__ import annotations

import subprocess
from pathlib import Path
from exports import finalize_master
from typing import Tuple

import numpy as np
import soundfile as sf
import pyloudnorm as pyln


TARGET_LUFS = -10.0
TARGET_SR = 44100


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio as float32 array shape (samples, channels)."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    return data, sr


def write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    sf.write(path, audio, sr, subtype="PCM_16")
    return path


def export_mp3(wav_path: str, mp3_path: str, bitrate: str = "320k") -> str:
    """Use ffmpeg if available; otherwise copy WAV bytes as a fallback."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-b:a", bitrate, mp3_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        Path(mp3_path).write_bytes(Path(wav_path).read_bytes())
    return mp3_path


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return audio
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def true_peak_limit(audio: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return np.clip(audio, -1.0, 1.0)
