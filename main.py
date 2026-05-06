"""
MixMasterOS background worker.

Polls the Supabase `audio_jobs` table, downloads source audio from the
`audio` storage bucket, processes it with the internal engine, then uploads
the mastered WAV/MP3 back to storage and marks the job complete.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import traceback
import subprocess
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client, Client

from audio_engine import (
    load_audio,
    write_wav,
    normalize_loudness,
    true_peak_limit,
)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
BUCKET = os.environ.get("MIXMASTER_BUCKET", "audio")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("[fatal] SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", flush=True)
    sys.exit(1)

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# --- helpers ---------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def storage_download(path: str, dest: str) -> None:
    data = sb.storage.from_(BUCKET).download(path)
    with open(dest, "wb") as f:
        f.write(data)


def storage_upload(path: str, src: str, content_type: str) -> None:
    with open(src, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path,
            f.read(),
            {"content-type": content_type, "upsert": "true"},
        )


def export_mp3(wav_path: str, mp3_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame",
         "-b:a", "320k", mp3_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def update_job(job_id: str, **fields) -> None:
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    sb.table("audio_jobs").update(fields).eq("id", job_id).execute()


def claim_next_job():
    res = (
        sb.table("audio_jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at")
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None
    job = rows[0]
    upd = (
        sb.table("audio_jobs")
        .update({
            "status": "processing",
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("id", job["id"])
        .eq("status", "queued")
        .execute()
    )
    if not upd.data:
        return None
    return job


# --- processing ------------------------------------------------------------

def process_master(job: dict) -> None:
    job_id = job["id"]
    src_path = job.get("source_path") or job.get("input_path")
    if not src_path:
        raise ValueError("job missing source_path")

    target_lufs = float(job.get("target_lufs") or -14.0)
    ceiling_db = float(job.get("ceiling_db") or -1.0)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.wav")
        out_wav = os.path.join(tmp, "out.wav")
        out_mp3 = os.path.join(tmp, "out.mp3")

        log(f"[{job_id}] downloading {src_path}")
        storage_download(src_path, in_path)

        log(f"[{job_id}] processing")
        samples, sr = load_audio(in_path)
        samples = normalize_loudness(samples, sr, target_lufs=target_lufs)
        samples = true_peak_limit(samples, ceiling_db=ceiling_db)
        write_wav(out_wav, samples, sr)
        export_mp3(out_wav, out_mp3)

        wav_key = f"processed/{job_id}.wav"
        mp3_key = f"processed/{job_id}.mp3"

        log(f"[{job_id}] uploading results")
        storage_upload(wav_key, out_wav, "audio/wav")
        storage_upload(mp3_key, out_mp3, "audio/mpeg")

        update_job(
            job_id,
            status="completed",
            output_wav_path=wav_key,
            output_mp3_path=mp3_key,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        log(f"[{job_id}] done")


def handle_job(job: dict) -> None:
    job_id = job["id"]
    try:
        process_master(job)
    except Exception as exc:
        log(f"[{job_id}] FAILED: {exc}")
        traceback.print_exc()
        try:
            update_job(job_id, status="failed", error=str(exc))
        except Exception:
            pass


# --- main loop -------------------------------------------------------------

def main() -> None:
    log("MixMasterOS worker started")
    while True:
        try:
            job = claim_next_job()
            if job:
                handle_job(job)
            else:
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log("shutting down")
            return
        except Exception as exc:
            log(f"poll error: {exc}")
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
