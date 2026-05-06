"""MixMasterOS internal DSP engine. No third-party mastering APIs."""
import numpy as np
import librosa
import pyloudnorm as pyln
import soundfile as sf


def load_audio(path: str, sr: int = 44100):
    audio, sample_rate = librosa.load(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    return audio.T, sample_rate


def save_audio(path: str, audio, sample_rate: int):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sample_rate)


def band_energy(audio, sr, low_hz, high_hz):
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    idx = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(fft[idx]))


def analyze_audio(path: str):
    audio, sr = load_audio(path)
    meter = pyln.Meter(sr)
    try:
        loudness = float(meter.integrated_loudness(audio))
    except Exception:
        loudness = -23.0
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    crest = float(peak / rms) if rms > 0 else 0.0
    mono = np.mean(audio, axis=1)
    return {
        "sample_rate": sr,
        "lufs": loudness,
        "peak": peak,
        "rms": rms,
        "crest_factor": crest,
        "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr))),
        "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=mono, sr=sr))),
        "low_energy": band_energy(mono, sr, 20, 150),
        "mud_energy": band_energy(mono, sr, 200, 450),
        "harsh_energy": band_energy(mono, sr, 2500, 5500),
        "air_energy": band_energy(mono, sr, 10000, 16000),
        "stereo_width": float(np.mean(np.abs(audio[:, 0] - audio[:, 1]))),
        "clipping_detected": bool(peak >= 0.98),
    }
