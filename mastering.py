"""
MixMasterOS internal worker. Polls Supabase for queued audio_jobs,
downloads source audio from the `audio` storage bucket, runs the private
DSP engine (pedalboard + librosa + pyloudnorm), uploads rendered masters
back to storage, and updates the project + job rows.

This worker is OURS. It is not a third-party mastering API. Users never
touch it. They just upload in the app and click Master.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
"""
import os
import sys
import time
import tempfile
import traceback
import subprocess

# Allow `python worker/main.py` execution: add this dir to sys.path so
# sibling modules (audio_engine, mastering, mixing) import cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client, Client

from audio_engine import analyze_audio
from mastering import master_track
from mixing import mix_stems

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
BUCKET = os.environ.get("AUDIO_BUCKET", "audio")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))

sb: Client = create_client(SUPABASE_URL, SERVICE_KEY)


def log(*a):
    print("[mixmaster-worker]", *a, flush=True)


def update_job(job_id, **fields):
    sb.table("audio_jobs").update(fields).eq("id", job_id).execute()


def update_project(project_id, **fields):
    sb.table("projects").update(fields).eq("id", project_id).execute()


def storage_download(path: str, dest: str):
    data = sb.storage.from_(BUCKET).download(path)
    with open(dest, "wb") as f:
        f.write(data)


def storage_upload(local_path: str, dest_path: str, content_type: str):
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            dest_path, f.read(),
            {"content-type": content_type, "upsert": "true"},
        )
    return dest_path


def export_mp3(input_wav: str, output_mp3: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_wav, "-codec:a", "libmp3lame", "-b:a", "320k", output_mp3],
        check=True, capture_output=True,
    )


def process_master(project, job_id):
    """Single-file mastering."""
    src_url = project.get("original_file_url")
    if not src_url:
        raise Exception("project has no original_file_url")

    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "input" + (os.path.splitext(src_url)[1] or ".wav"))
        wav_path = os.path.join(td, "master.wav")
        mp3_path = os.path.join(td, "master.mp3")

        update_job(job_id, progress=15, stage="Downloading source")
        storage_download(src_url, in_path)

        update_job(job_id, progress=30, stage="Analyzing waveform")
        before = analyze_audio(in_path)

        update_job(job_id, progress=55, stage="Applying mastering chain")
        settings = master_track(
            in_path, wav_path,
            project.get("genre") or "custom",
            project.get("sound_profile") or "warm-polished",
            target_lufs=None,
        )

        update_job(job_id, progress=75, stage="Encoding MP3")
        export_mp3(wav_path, mp3_path)

        update_job(job_id, progress=88, stage="Final analysis")
        after = analyze_audio(wav_path)

        user_id = project["user_id"]
        pid = project["id"]
        wav_remote = f"{user_id}/{pid}/master.wav"
        mp3_remote = f"{user_id}/{pid}/master.mp3"

        update_job(job_id, progress=94, stage="Uploading master")
        storage_upload(wav_path, wav_remote, "audio/wav")
        storage_upload(mp3_path, mp3_remote, "audio/mpeg")

        update_project(pid,
            status="complete",
            master_wav_url=wav_remote,
            master_mp3_url=mp3_remote,
            lufs_target=settings["target_lufs"],
            peak_level=after.get("peak"),
            analysis_json={"before": before, "after": after, "settings": settings, "engine": "mixmasteros-internal-v1"},
        )


def process_stem_workflow(project, job_id):
    """Mix stems then master."""
    stems_json = project.get("stems_json") or {}
    stem_paths = stems_json if isinstance(stems_json, dict) else {}
    if not stem_paths:
        raise Exception("project has no stems_json")

    with tempfile.TemporaryDirectory() as td:
        local_stems = {}
        update_job(job_id, progress=10, stage="Downloading stems")
        for stem_type, remote in stem_paths.items():
            if not remote:
                continue
            ext = os.path.splitext(remote)[1] or ".wav"
            local = os.path.join(td, f"{stem_type}{ext}")
            storage_download(remote, local)
            local_stems[stem_type] = local

        mix_path = os.path.join(td, "premix.wav")
        wav_path = os.path.join(td, "master.wav")
        mp3_path = os.path.join(td, "master.mp3")

        update_job(job_id, progress=40, stage="Mixing stems")
        mix_stems(local_stems, mix_path, project.get("genre") or "custom")

        update_job(job_id, progress=65, stage="Mastering mix")
        before = analyze_audio(mix_path)
        settings = master_track(mix_path, wav_path,
                                project.get("genre") or "custom",
                                project.get("sound_profile") or "warm-polished")

        update_job(job_id, progress=82, stage="Encoding MP3")
        export_mp3(wav_path, mp3_path)

        after = analyze_audio(wav_path)

        user_id = project["user_id"]
        pid = project["id"]
        wav_remote = f"{user_id}/{pid}/master.wav"
        mp3_remote = f"{user_id}/{pid}/master.mp3"

        update_job(job_id, progress=94, stage="Uploading master")
        storage_upload(wav_path, wav_remote, "audio/wav")
        storage_upload(mp3_path, mp3_remote, "audio/mpeg")

        update_project(pid,
            status="complete",
            master_wav_url=wav_remote,
            master_mp3_url=mp3_remote,
            lufs_target=settings["target_lufs"],
            peak_level=after.get("peak"),
            analysis_json={"before_master": before, "after_master": after,
                           "settings": settings, "stems_used": list(local_stems.keys()),
                           "engine": "mixmasteros-internal-v1"},
        )


def claim_next_job():
    res = sb.table("audio_jobs").select("*").eq("status", "queued") \
        .order("created_at").limit(1).execute()
    rows = res.data or []
    if not rows:
        return None
    job = rows[0]
    upd = sb.table("audio_jobs").update({
        "status": "processing", "started_at": "now()", "progress": 5, "stage": "Claimed"
    }).eq("id", job["id"]).eq("status", "queued").execute()
    if not upd.data:
        return None
    return upd.data[0]


def handle_job(job):
    project_id = job["project_id"]
    proj = sb.table("projects").select("*").eq("id", project_id).single().execute().data
    if not proj:
        update_job(job["id"], status="failed", error_message="project not found")
        return

    workflow = proj.get("workflow") or "master"
    log(f"job {job['id']} project {project_id} workflow={workflow}")

    try:
        if workflow in ("master", "master-only"):
            process_master(proj, job["id"])
        else:
            process_stem_workflow(proj, job["id"])

        update_job(job["id"], status="complete", progress=100,
                   stage="Complete", completed_at="now()")

        prof = sb.table("profiles").select("credits_remaining").eq("id", proj["user_id"]).single().execute().data
        if prof and (prof.get("credits_remaining") or 0) > 0:
            sb.table("profiles").update({
                "credits_remaining": prof["credits_remaining"] - 1
            }).eq("id", proj["user_id"]).execute()

    except Exception as e:
        log("FAILED", e)
        traceback.print_exc()
        update_job(job["id"], status="failed",
                   error_message=str(e)[:500],
                   completed_at="now()")
        update_project(project_id, status="failed")


def main():
    log("MixMasterOS internal worker started. Polling every", POLL_INTERVAL, "s")
    while True:
        try:
            job = claim_next_job()
            if job:
                handle_job(job)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            log("loop error:", e)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
