"""
MixMasterOS internal audio engine.
All DSP primitives used by main.py live here.

Public API:
    load_audio(path)            -> (samples: np.ndarray [channels, frames], sr: int)
    write_wav(path, samples, sr) -> None
    normalize_loudness(samples, sr, target_lufs=-14.0) -> np.ndarray
    true_peak_limit(samples, ceiling_db=-1.0) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import pyloudnorm as pyln


def load_audio(path: str):
    """Load an audio file as float32 [channels, frames] and sample rate."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    # soundfile returns [frames, channels]; convert to [channels, frames]
    samples = data.T.copy()
    return samples, int(sr)


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    """Write a [channels, frames] float array as 24-bit WAV."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        out = arr.reshape(-1, 1)
    else:
        out = arr.T  # back to [frames, channels]
    sf.write(path, out, sr, subtype="PCM_24")


def normalize_loudness(samples: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Normalize integrated loudness to target LUFS using ITU-R BS.1770."""
    arr = np.asarray(samples, dtype=np.float32)
    # pyloudnorm expects [frames] mono or [frames, channels] stereo
    if arr.ndim == 1:
        meter_input = arr
    else:
        meter_input = arr.T

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(meter_input)
    except Exception:
        return arr

    if not np.isfinite(loudness):
        return arr

    normalized = pyln.normalize.loudness(meter_input, loudness, target_lufs)
    if normalized.ndim == 1:
        return normalized.astype(np.float32)
    return normalized.T.astype(np.float32)


def true_peak_limit(samples: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Simple peak limiter: scales signal so its peak sits at ceiling_db dBFS."""
    arr = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 0.0:
        return arr

    ceiling_linear = 10.0 ** (ceiling_db / 20.0)
    if peak <= ceiling_linear:
        return arr

    gain = ceiling_linear / peak
    return (arr * gain).astype(np.float32)
