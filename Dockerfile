import numpy as np
from pedalboard import Pedalboard, Compressor, Gain, HighpassFilter, LowpassFilter, Reverb
from audio_engine import load_audio, save_audio


HP = {"lead_vocal": 80, "background_vocals": 100, "adlibs": 100,
      "guitars": 70, "keys": 60, "drums": 30, "bass": 25, "beat": 25}
LP = {"bass": 6000, "drums": 16000, "beat": 18000, "lead_vocal": 17000}


def stem_settings(genre: str):
    if genre in ["rap", "trap"]:
        return {"lead_vocal": {"gain": 2.0, "comp": 2.8}, "adlibs": {"gain": -2.5, "comp": 2.0},
                "beat": {"gain": -1.0, "comp": 1.6}, "drums": {"gain": 1.0, "comp": 2.2},
                "bass": {"gain": 1.5, "comp": 2.0}}
    if genre in ["corrido", "christian_corrido"]:
        return {"lead_vocal": {"gain": 1.8, "comp": 2.0}, "guitars": {"gain": 0.8, "comp": 1.4},
                "bass": {"gain": 0.5, "comp": 1.6}, "drums": {"gain": 0.6, "comp": 1.8}}
    if genre in ["worship", "gospel"]:
        return {"lead_vocal": {"gain": 2.0, "comp": 1.8}, "background_vocals": {"gain": -1.0, "comp": 1.5},
                "keys": {"gain": 0.5, "comp": 1.3}, "guitars": {"gain": 0.2, "comp": 1.3},
                "drums": {"gain": 0.3, "comp": 1.5}, "bass": {"gain": 0.2, "comp": 1.5}}
    return {"lead_vocal": {"gain": 1.5, "comp": 2.0}, "beat": {"gain": -0.5, "comp": 1.5}}


def process_stem(audio, sr, stem_type, settings):
    cfg = settings.get(stem_type, {})
    plugins = [
        HighpassFilter(cutoff_frequency_hz=HP.get(stem_type, 30)),
        LowpassFilter(cutoff_frequency_hz=LP.get(stem_type, 18000)),
        Compressor(threshold_db=-20, ratio=cfg.get("comp", 1.5), attack_ms=10, release_ms=100),
        Gain(gain_db=cfg.get("gain", 0)),
    ]
    if stem_type in ["lead_vocal", "background_vocals", "adlibs"]:
        plugins.append(Reverb(room_size=0.08, wet_level=0.04, dry_level=0.96))
    return Pedalboard(plugins)(audio, sr)


def mix_stems(stem_paths: dict, output_path: str, genre: str):
    settings = stem_settings(genre)
    mixed = None
    sr_used = 44100
    for stem_type, path in stem_paths.items():
        audio, sr = load_audio(path)
        sr_used = sr
        processed = process_stem(audio, sr, stem_type, settings)
        if mixed is None:
            mixed = processed
        else:
            n = min(len(mixed), len(processed))
            mixed = mixed[:n] + processed[:n]
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.85
    save_audio(output_path, mixed, sr_used)
    return output_path
