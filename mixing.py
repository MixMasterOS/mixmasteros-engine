"""Mastering chain: EQ → compression → limiter → loudness normalization."""
from __future__ import annotations

import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, Limiter

from audio_engine import load_audio, write_wav, normalize_loudness, true_peak_limit, TARGET_LUFS


def master_file(input_path: str, output_path: str, target_lufs: float = TARGET_LUFS) -> str:
    audio, sr = load_audio(input_path)

    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=30.0),
        LowShelfFilter(cutoff_frequency_hz=120.0, gain_db=1.5, q=0.7),
        HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=2.0, q=0.7),
        Compressor(threshold_db=-14.0, ratio=2.0, attack_ms=10.0, release_ms=120.0),
        Limiter(threshold_db=-1.0, release_ms=80.0),
    ])

    processed = board(audio, sr)
    processed = normalize_loudness(np.asarray(processed, dtype=np.float32), sr, target_lufs)
    processed = true_peak_limit(processed, ceiling_db=-1.0)

    write_wav(output_path, processed, sr)
    return output_path
