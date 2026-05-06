"""MixMasterOS background worker.

Polls the Supabase `audio_jobs` table, claims queued jobs, runs the internal
mixing/mastering DSP chain, uploads results back to Supabase Storage, and
updates job + project status. No public HTTP API. No third-party mastering.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import tempfile
import traceback
from pathlib import Path

# Make sibling modules importable when run as `python worker/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from supabase import create_client, Client

import storage
from audio_engine import export_mp3
from mastering import master_file
from mixing import mix_stems

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "5"))
WORKER_ID = os.environ.get("WORKER_ID") or f"worker-{uuid.uuid4().hex[:8]}"

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# ---------- Job lifecycle helpers ----------

def claim_next_job():
    """Atomically claim the oldest queued job by flipping it to processing."""
    res = sb.table("audio_jobs").select("*").eq("status", "queued") \
        .order("created_at", desc=False).limit(1).execute()
    rows = res.data or []
    if not rows:
        return None
    job = rows[0]
    upd = sb.table("audio_jobs").update({
        "status": "processing",
        "stage": "Starting",
        "progress": 5,
        "worker_id": WORKER_ID,
    }).eq("id", job["id"]).eq("status", "queued").execute()
    if not upd.data:
        # Another worker grabbed it first.
        return None
    return upd.data[0]


def update_job(job_id: str, **fields):
    sb.table("audio_jobs").update(fields).eq("id", job_id).execute()


def fail_job(job_id: str, project_id: str | None, message: str):
    update_job(job_id, status="failed", stage="Failed", error_message=message[:1000])
    if project_id:
        sb.table("projects").update({"status": "failed"}).eq("id", project_id).execute()


def complete_job(job_id: str, project_id: str, wav_path: str, mp3_path: str):
    update_job(
        job_id,
        status="completed",
        stage="Done",
        progress=100,
        output_wav_path=wav_path,
        output_mp3_path=mp3_path,
    )
    sb.table("projects").update({
        "status": "completed",
        "mastered_wav_path": wav_path,
        "mastered_mp3_path": mp3_path,
    }).eq("id", project_id).execute()


# ---------- Processing ----------

def _decrement_credit(user_id: str):
    try:
        prof = sb.table("profiles").select("credits_remaining").eq("id", user_id).single().execute()
        remaining = (prof.data or {}).get("credits_remaining", 0) or 0
        if remaining > 0:
            sb.table("profiles").update({"credits_remaining": remaining - 1}).eq("id", user_id).execute()
    except Exception as e:
        print(f"[warn] credit decrement failed: {e}")


def process_master(job: dict, project: dict, work_dir: str):
    src_path = project.get("source_audio_path")
    if not src_path:
        raise RuntimeError("Project has no source_audio_path")

    update_job(job["id"], stage="Downloading source", progress=15)
    local_in = os.path.join(work_dir, "input" + Path(src_path).suffix)
    storage.download(sb, src_path, local_in)

    update_job(job["id"], stage="Mastering", progress=45)
    local_wav = os.path.join(work_dir, "mastered.wav")
    master_file(local_in, local_wav)

    update_job(job["id"], stage="Encoding MP3", progress=75)
    local_mp3 = os.path.join(work_dir, "mastered.mp3")
    export_mp3(local_wav, local_mp3)

    update_job(job["id"], stage="Uploading", progress=90)
    base = f"{project['user_id']}/{project['id']}"
    wav_dest = f"{base}/mastered.wav"
    mp3_dest = f"{base}/mastered.mp3"
    storage.upload(sb, local_wav, wav_dest, "audio/wav")
    storage.upload(sb, local_mp3, mp3_dest, "audio/mpeg")

    return wav_dest, mp3_dest


def process_mix_master(job: dict, project: dict, work_dir: str):
    stems = project.get("stems") or {}
    if not stems:
        # Fall back to single-file master if no stems were supplied.
        return process_master(job, project, work_dir)

    update_job(job["id"], stage="Downloading stems", progress=15)
    local_stems: dict[str, str] = {}
    for kind, remote in stems.items():
        if not remote:
            continue
        local = os.path.join(work_dir, f"stem_{kind}{Path(remote).suffix}")
        storage.download(sb, remote, local)
        local_stems[kind] = local

    update_job(job["id"], stage="Mixing stems", progress=40)
    local_mix = os.path.join(work_dir, "mix.wav")
    mix_stems(local_stems, local_mix)

    update_job(job["id"], stage="Mastering", progress=65)
    local_wav = os.path.join(work_dir, "mastered.wav")
    master_file(local_mix, local_wav)

    update_job(job["id"], stage="Encoding MP3", progress=80)
    local_mp3 = os.path.join(work_dir, "mastered.mp3")
    export_mp3(local_wav, local_mp3)

    update_job(job["id"], stage="Uploading", progress=92)
    base = f"{project['user_id']}/{project['id']}"
    wav_dest = f"{base}/mastered.wav"
    mp3_dest = f"{base}/mastered.mp3"
    storage.upload(sb, local_wav, wav_dest, "audio/wav")
    storage.upload(sb, local_mp3, mp3_dest, "audio/mpeg")
    return wav_dest, mp3_dest


def handle_job(job: dict):
    project_id = job.get("project_id")
    try:
        proj_res = sb.table("projects").select("*").eq("id", project_id).single().execute()
        project = proj_res.data
        if not project:
            raise RuntimeError(f"Project {project_id} not found")

        with tempfile.TemporaryDirectory(prefix="mmos_") as work_dir:
            if (job.get("job_type") or "master") == "master":
                wav_path, mp3_path = process_master(job, project, work_dir)
            else:
                wav_path, mp3_path = process_mix_master(job, project, work_dir)

        complete_job(job["id"], project_id, wav_path, mp3_path)
        _decrement_credit(job["user_id"])
        print(f"[ok] job {job['id']} completed")
    except Exception as e:
        traceback.print_exc()
        fail_job(job["id"], project_id, str(e))
        print(f"[err] job {job['id']} failed: {e}")


def main():
    print(f"MixMasterOS worker {WORKER_ID} started; polling every {POLL_INTERVAL}s")
    while True:
        try:
            job = claim_next_job()
            if job:
                print(f"[claim] job {job['id']} ({job.get('job_type')})")
                handle_job(job)
            else:
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("Shutting down")
            return
        except Exception as e:
            traceback.print_exc()
            print(f"[loop-err] {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
