"""
exports.py — Final-stage deliverable renderer for MixMasterOS engine.

Takes the already-mastered float source (post EQ/comp/sat/width/soft-clip/TP
limiter/loudness norm) and fans it out into 5 deliverables:

  - 32-bit float / 96 kHz WAV  (archival, no dither)
  - 24-bit PCM   / 96 kHz WAV  (studio HD, soxr, no dither)
  - 16-bit PCM   / 44.1 kHz WAV (streaming, soxr + TPDF dither)
  - 320 kbps CBR MP3            (LAME -b:a 320k -q 0 joint stereo + reservoir)
  - V0 VBR MP3 (~245 kbps)      (LAME -q:a 0 -q 0)

Then uploads each to Supabase Storage and PATCHes the project row, and
finally calls the verify-wav edge function.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  FFMPEG_BIN           (default: "ffmpeg", must be built with libmp3lame + libsoxr)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET = os.environ.get("AUDIO_BUCKET", "audio")


# ---------------------------------------------------------------------------
# FFmpeg render
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> None:
    """Run ffmpeg, raise with stderr on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({res.returncode})\nCMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}"
        )


@dataclass
class RenderedSet:
    wav32f: Path
    wav24:  Path
    wav16:  Path
    mp3_320: Path
    mp3_v0: Path


def render_deliverables(master_src: Path, out_dir: Path, basename: str) -> RenderedSet:
    """
    master_src: path to the already-mastered audio (any format ffmpeg can read).
                Best is a 32-bit float WAV written by the DSP stage.
    out_dir:    directory where the 5 deliverables will be written.
    basename:   safe filename stem, e.g. "my_song_master".
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    wav32f  = out_dir / f"{basename}_32f_96k.wav"
    wav24   = out_dir / f"{basename}_24b_96k.wav"
    wav16   = out_dir / f"{basename}_16b_441.wav"
    mp3_320 = out_dir / f"{basename}_320cbr.mp3"
    mp3_v0  = out_dir / f"{basename}_v0vbr.mp3"

    common_meta = ["-metadata", "encoded_by=FNLZ"]

    # 32-bit float / 96 kHz — archival, no dither, no SR change beyond resample if needed
    _run([
        FFMPEG, "-y", "-i", str(master_src),
        "-ar", "96000", "-ac", "2",
        "-c:a", "pcm_f32le",
        "-rf64", "auto",
        *common_meta,
        str(wav32f),
    ])

    # 24-bit / 96 kHz — soxr precision 33, no dither (24-bit headroom is enough)
    _run([
        FFMPEG, "-y", "-i", str(master_src),
        "-ar", "96000", "-ac", "2",
        "-af", "aresample=resampler=soxr:precision=33",
        "-c:a", "pcm_s24le",
        *common_meta,
        str(wav24),
    ])

    # 16-bit / 44.1 kHz — soxr + TPDF (triangular_hp) dither
    _run([
        FFMPEG, "-y", "-i", str(master_src),
        "-ar", "44100", "-ac", "2",
        "-af", "aresample=resampler=soxr:precision=33:dither_method=triangular_hp",
        "-c:a", "pcm_s16le",
        *common_meta,
        str(wav16),
    ])

    # MP3 320 CBR — joint stereo, bit reservoir, max LAME quality (slowest = best)
    _run([
        FFMPEG, "-y", "-i", str(master_src),
        "-ar", "44100", "-ac", "2",
        "-c:a", "libmp3lame",
        "-b:a", "320k",
        "-compression_level", "0",
        "-joint_stereo", "1",
        "-reservoir", "1",
        *common_meta,
        str(mp3_320),
    ])

    # MP3 V0 VBR (~245 kbps)
    _run([
        FFMPEG, "-y", "-i", str(master_src),
        "-ar", "44100", "-ac", "2",
        "-c:a", "libmp3lame",
        "-q:a", "0",
        "-compression_level", "0",
        *common_meta,
        str(mp3_v0),
    ])

    return RenderedSet(wav32f, wav24, wav16, mp3_320, mp3_v0)


# ---------------------------------------------------------------------------
# Supabase upload + DB patch
# ---------------------------------------------------------------------------

def _storage_upload(local: Path, storage_path: str, content_type: str) -> str:
    """Upload to Supabase Storage (service-role). Returns the storage path."""
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{storage_path}"
    with open(local, "rb") as f:
        r = requests.post(
            url,
            data=f,
            headers={
                "Authorization": f"Bearer {SERVICE_KEY}",
                "apikey": SERVICE_KEY,
                "Content-Type": content_type,
                "x-upsert": "true",
            },
            timeout=600,
        )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"storage upload failed {r.status_code}: {r.text}")
    return storage_path


def _patch_project(project_id: str, patch: dict) -> None:
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/projects?id=eq.{project_id}",
        json=patch,
        headers={
            "Authorization": f"Bearer {SERVICE_KEY}",
            "apikey": SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if r.status_code not in (200, 204):
        raise RuntimeError(f"project patch failed {r.status_code}: {r.text}")


def _lame_version() -> str:
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-version"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            if "libmp3lame" in line.lower():
                return line.strip()
    except Exception:
        pass
    return "libmp3lame"


def upload_and_patch(project_id: str, user_id: str, rendered: RenderedSet) -> dict:
    """Upload all 5 files and PATCH the project row. Returns the patch dict applied."""
    base = f"{user_id}/{project_id}/master"

    paths = {
        "master_wav32f_url":  _storage_upload(rendered.wav32f,  f"{base}/master_32f_96k.wav",  "audio/wav"),
        "master_wav_url":     _storage_upload(rendered.wav24,   f"{base}/master_24b_96k.wav",  "audio/wav"),
        "master_wav16_url":   _storage_upload(rendered.wav16,   f"{base}/master_16b_441.wav",  "audio/wav"),
        "master_mp3_320_url": _storage_upload(rendered.mp3_320, f"{base}/master_320cbr.mp3",   "audio/mpeg"),
        "master_mp3_v0_url":  _storage_upload(rendered.mp3_v0,  f"{base}/master_v0vbr.mp3",    "audio/mpeg"),
    }
    # legacy single-MP3 column → point at 320 CBR for backward compat
    paths["master_mp3_url"] = paths["master_mp3_320_url"]

    patch = {
        **paths,
        "mp3_encoder_version": _lame_version(),
        "status": "complete",
    }
    _patch_project(project_id, patch)
    return patch


# ---------------------------------------------------------------------------
# verify-wav invocation
# ---------------------------------------------------------------------------

def call_verify_wav(project_id: str) -> None:
    """Fire-and-forget call to the verify-wav edge function."""
    try:
        r = requests.post(
            f"{SUPABASE_URL}/functions/v1/verify-wav",
            json={"projectId": project_id},
            headers={
                "Authorization": f"Bearer {SERVICE_KEY}",
                "apikey": SERVICE_KEY,
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        if r.status_code >= 400:
            print(f"[verify-wav] non-fatal: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[verify-wav] non-fatal exception: {e}")


# ---------------------------------------------------------------------------
# One-shot helper used by mastering.py
# ---------------------------------------------------------------------------

def finalize_master(
    project_id: str,
    user_id: str,
    mastered_source_path: Path,
    project_name: str,
    work_dir: Optional[Path] = None,
) -> dict:
    """
    Renders 5 deliverables, uploads them, patches the row, calls verify-wav.
    Returns the patch dict that was applied.
    """
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="fnlz_export_"))
        cleanup = True

    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in project_name) or "master"

    try:
        rendered = render_deliverables(mastered_source_path, work_dir, safe)
        patch = upload_and_patch(project_id, user_id, rendered)
        call_verify_wav(project_id)
        return patch
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)
