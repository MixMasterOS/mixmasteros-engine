"""Supabase Storage helpers for the MixMasterOS worker."""
from __future__ import annotations

import os
from typing import Optional

BUCKET = "audio"


def download(supabase, path: str, dest: str) -> str:
    """Download an object from the audio bucket to a local file."""
    data = supabase.storage.from_(BUCKET).download(path)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def upload(supabase, local_path: str, dest_path: str, content_type: str) -> str:
    """Upload a local file to the audio bucket. Overwrites if exists."""
    with open(local_path, "rb") as f:
        data = f.read()
    try:
        supabase.storage.from_(BUCKET).upload(
            path=dest_path,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    except Exception:
        # Fallback for older client signatures.
        supabase.storage.from_(BUCKET).update(
            path=dest_path, file=data,
            file_options={"content-type": content_type},
        )
    return dest_path


def signed_url(supabase, path: str, expires_in: int = 60 * 60 * 24 * 7) -> Optional[str]:
    try:
        res = supabase.storage.from_(BUCKET).create_signed_url(path, expires_in)
        return res.get("signedURL") or res.get("signed_url")
    except Exception:
        return None
