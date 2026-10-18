from pathlib import Path
import uuid
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://aannvqyrjjzimosdvhoh.supabase.co")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SECRET_KEY")

_supabase = None

def get_supabase():
    global _supabase
    if _supabase is None:
        if not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_SERVICE_KEY or SUPABASE_SECRET_KEY not set")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase

BUCKET = "audio-files"

def upload_audio_file(local_path: Path, user_id: str = None, folder: str = "uploads") -> dict:
    supabase = get_supabase()
    
    file_name = f"{folder}/{uuid.uuid4().hex}_{local_path.name}"
    
    with local_path.open("rb") as f:
        result = supabase.storage.from_(BUCKET).upload(
            file_name,
            f,
            file_options={"content-type": f"audio/{local_path.suffix[1:]}", "upsert": "false"}
        )
    
    if hasattr(result, 'error') and result.error:
        raise RuntimeError(f"Upload failed: {result.error}")
    
    public_url = supabase.storage.from_(BUCKET).get_public_url(file_name)
    
    return {
        "storage_path": file_name,
        "storage_bucket": BUCKET,
        "public_url": public_url,
        "file_size": local_path.stat().st_size
    }

def delete_audio_file(storage_path: str) -> bool:
    supabase = get_supabase()
    result = supabase.storage.from_(BUCKET).remove([storage_path])
    return not (hasattr(result, 'error') and result.error)

def create_signed_url(storage_path: str, expires_in: int = 3600) -> str:
    supabase = get_supabase()
    result = supabase.storage.from_(BUCKET).create_signed_url(storage_path, expires_in)
    if hasattr(result, 'error') and result.error:
        raise RuntimeError(f"Signed URL failed: {result.error}")
    return result.get('signedURL', '')

def cleanup_temp_file(file_path: Path):
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

def cleanup_temp_files(file_paths: list):
    for p in file_paths:
        cleanup_temp_file(p)