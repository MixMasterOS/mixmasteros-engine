# MixMasterOS Worker

Private Supabase background worker for MixMasterOS.

## Deploy on Render (Background Worker)

1. Push this folder to GitHub.
2. New → Background Worker → connect repo.
3. Environment: Docker.
4. Add env vars:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
5. Deploy.

## Files

- `Dockerfile` — Python 3.11 + ffmpeg + libsndfile
- `requirements.txt` — Python deps
- `main.py` — polling loop + job handler
- `audio_engine.py` — `load_audio`, `write_wav`, `normalize_loudness`, `true_peak_limit`
