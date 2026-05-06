"""
MixMasterOS background worker.

Polls the Supabase `audio_jobs` table, fetches the matching project row,
downloads source audio from the `audio` storage bucket, processes it with
the internal engine, then uploads the mastered WAV/MP3 back to storage and
updates both the job and the project.
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

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
BUCKET = os.environ.get("MIXMASTER_BUCKET", "audio").strip()
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("[fatal] SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set", flush=True)
    sys.exit(1)

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# --- helpers ---------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    sb.table("audio_jobs").update(fields).eq("id", job_id).execute()


def update_project(project_id: str, **fields) -> None:
    sb.table("projects").update(fields).eq("id", project_id).execute()


def fetch_project(project_id: str) -> dict | None:
    res = (
        sb.table("projects")
        .select("*")
        .eq("id", project_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def claim_next_job():
    """Atomically claim the next queued job. Verbose logging on every poll."""
    log("checking for queued jobs (status='queued')")
    try:
        res = (
            sb.table("audio_jobs")
            .select("id,project_id,user_id,status,stage,progress,created_at")
            .eq("status", "queued")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        log(f"[poll] supabase select FAILED: {exc}")
        traceback.print_exc()
        return None

    rows = res.data or []
    log(f"[poll] supabase returned {len(rows)} row(s): {rows}")

    if not rows:
        return None

    job = rows[0]
    job_id = job["id"]
    log(f"[poll] attempting to claim job {job_id}")

    try:
        upd = (
            sb.table("audio_jobs")
            .update({
                "status": "processing",
                "started_at": now_iso(),
                "stage": "Processing",
                "progress": 5,
            })
            .eq("id", job_id)
            .eq("status", "queued")
            .execute()
        )
    except Exception as exc:
        log(f"[poll] claim update FAILED for job {job_id}: {exc}")
        traceback.print_exc()
        return None

    if not upd.data:
        log(f"[poll] job {job_id} was claimed by another worker")
        return None

    log(f"[poll] CLAIMED job {job_id} -> status='processing'")
    return upd.data[0]


# --- processing ------------------------------------------------------------
def normalize_storage_path(value: str) -> str:
    if not value:
        return value
    marker = f"/object/public/{BUCKET}/"
    if marker in value:
        return value.split(marker, 1)[1].split("?", 1)[0]
    marker2 = f"/object/sign/{BUCKET}/"
    if marker2 in value:
        return value.split(marker2, 1)[1].split("?", 1)[0]
    marker3 = f"/{BUCKET}/"
    if marker3 in value:
        return value.split(marker3, 1)[1].split("?", 1)[0]
    return value


def process_master(job: dict) -> None:
    job_id = job["id"]
    project_id = job["project_id"]

    project = fetch_project(project_id)
    if not project:
        raise RuntimeError(f"project {project_id} not found")

    src = project.get("original_file_url")
    if not src:
        raise RuntimeError("project missing original_file_url")
    src_key = normalize_storage_path(src)

    target_lufs = float(project.get("lufs_target") or -14.0)
    ceiling_db = float(project.get("peak_level") or -1.0)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "in.wav")
        out_wav = os.path.join(tmp, "out.wav")
        out_mp3 = os.path.join(tmp, "out.mp3")

        log(f"[{job_id}] downloading {src_key}")
        update_job(job_id, stage="Downloading", progress=15)
        storage_download(src_key, in_path)

        log(f"[{job_id}] processing")
        update_job(job_id, stage="Mastering", progress=45)
        samples, sr = load_audio(in_path)
        samples = normalize_loudness(samples, sr, target_lufs=target_lufs)
        samples = true_peak_limit(samples, ceiling_db=ceiling_db)
        write_wav(out_wav, samples, sr)
        export_mp3(out_wav, out_mp3)

        wav_key = f"masters/{project_id}/{job_id}.wav"
        mp3_key = f"masters/{project_id}/{job_id}.mp3"

        log(f"[{job_id}] uploading results")
        update_job(job_id, stage="Uploading", progress=85)
        storage_upload(wav_key, out_wav, "audio/wav")
        storage_upload(mp3_key, out_mp3, "audio/mpeg")

        update_project(
            project_id,
            master_wav_url=wav_key,
            master_mp3_url=mp3_key,
            status="complete",
            lufs_target=target_lufs,
            peak_level=ceiling_db,
        )
        update_job(
            job_id,
            status="complete",
            stage="Complete",
            progress=100,
            completed_at=now_iso(),
        )

        try:
            user_id = job.get("user_id")
            if user_id:
                prof = (
                    sb.table("profiles")
                    .select("credits_remaining")
                    .eq("id", user_id)
                    .limit(1)
                    .execute()
                )
                if prof.data:
                    remaining = max(0, int(prof.data[0].get("credits_remaining") or 0) - 1)
                    sb.table("profiles").update(
                        {"credits_remaining": remaining}
                    ).eq("id", user_id).execute()
        except Exception as e:
            log(f"[{job_id}] credit decrement skipped: {e}")

        log(f"[{job_id}] done")


def handle_job(job: dict) -> None:
    job_id = job["id"]
    project_id = job.get("project_id")
    try:
        process_master(job)
    except Exception as exc:
        log(f"[{job_id}] FAILED: {exc}")
        traceback.print_exc()
        try:
            update_job(
                job_id,
                status="failed",
                stage="Failed",
                error_message=str(exc)[:500],
                completed_at=now_iso(),
            )
        except Exception:
            pass
        if project_id:
            try:
                update_project(project_id, status="failed")
            except Exception:
                pass


# --- main loop -------------------------------------------------------------
def main() -> None:
    log(f"MixMasterOS worker started — url={SUPABASE_URL} bucket={BUCKET} interval={POLL_INTERVAL}s")
    cycle = 0
    while True:
        cycle += 1
        try:
            log(f"--- poll cycle #{cycle} ---")
            job = claim_next_job()
            if job:
                handle_job(job)
                # Immediately loop again to drain the queue.
                continue
            else:
                log(f"no queued jobs, sleeping {POLL_INTERVAL}s")
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log("shutting down")
            return
        except Exception as exc:
            log(f"[poll] UNHANDLED loop error: {exc}")
            traceback.print_exc()
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[fatal] worker crashed: {exc}")
        traceback.print_exc()
        # Re-raise so Render restarts the service.
        raise
