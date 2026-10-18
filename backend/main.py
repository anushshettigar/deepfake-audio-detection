from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json
import traceback
import subprocess
import time
import os
from collections import defaultdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

load_dotenv()

from detector.detector import predict_audio
from storage import (
    upload_audio_file,
    cleanup_temp_file,
    cleanup_temp_files,
    get_supabase,
)


app = FastAPI(
    title="DeepShield API",
    version="1.0.0"
)


DIST_DIR = Path(__file__).resolve().parent.parent / "dist"

if DIST_DIR.exists():
    app.mount(
        "/dist",
        StaticFiles(directory=str(DIST_DIR)),
        name="dist"
    )


SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    ""
).rstrip("/")

SUPABASE_JWT_SECRET = os.environ.get(
    "SUPABASE_JWT_SECRET"
)

SUPABASE_JWKS_URL = os.environ.get(
    "SUPABASE_JWKS_URL"
)

if not SUPABASE_JWKS_URL and SUPABASE_URL:
    SUPABASE_JWKS_URL = (
        f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    )


SUPABASE_ISSUER = (
    f"{SUPABASE_URL}/auth/v1"
    if SUPABASE_URL
    else None
)

RATE_LIMIT_REQUESTS = int(
    os.environ.get(
        "RATE_LIMIT_REQUESTS",
        "300"
    )
)

RATE_LIMIT_WINDOW = int(
    os.environ.get(
        "RATE_LIMIT_WINDOW",
        "60"
    )
)


rate_limit_store = defaultdict(list)

security = HTTPBearer(
    auto_error=False
)


if (
    SUPABASE_JWT_SECRET
    and SUPABASE_JWT_SECRET.strip()
    == "your-jwt-secret-here"
):
    SUPABASE_JWT_SECRET = None


_jwks_client = None


if SUPABASE_JWKS_URL:

    _jwks_client = jwt.PyJWKClient(
        SUPABASE_JWKS_URL
    )

elif SUPABASE_JWT_SECRET:

    pass

else:

    raise RuntimeError(
        "Neither SUPABASE_JWKS_URL nor "
        "SUPABASE_JWT_SECRET is configured."
    )


def decode_supabase_token(
    token: str
):

    if not token:

        raise jwt.InvalidTokenError(
            "Empty JWT token"
        )

    try:

        header = jwt.get_unverified_header(
            token
        )

    except jwt.InvalidTokenError:

        raise

    algorithm = header.get(
        "alg"
    )

    if not algorithm:

        raise jwt.InvalidTokenError(
            "JWT does not contain an alg header"
        )

    if _jwks_client:

        if algorithm != "ES256":

            raise jwt.InvalidAlgorithmError(
                "ES256 JWT expected when "
                "Supabase JWKS verification is configured"
            )

        signing_key = (
            _jwks_client
            .get_signing_key_from_jwt(
                token
            )
            .key
        )

        decode_options = {
            "algorithms": ["ES256"],
            "audience": "authenticated"
        }

        if SUPABASE_ISSUER:

            decode_options["issuer"] = SUPABASE_ISSUER

        return jwt.decode(
            token,
            signing_key,
            **decode_options
        )

    if algorithm != "HS256":

        raise jwt.InvalidAlgorithmError(
            "HS256 JWT expected when "
            "SUPABASE_JWT_SECRET is used "
            "without JWKS."
        )

    decode_options = {
        "algorithms": ["HS256"],
        "audience": "authenticated"
    }

    if SUPABASE_ISSUER:

        decode_options["issuer"] = SUPABASE_ISSUER

    return jwt.decode(
        token,
        SUPABASE_JWT_SECRET,
        **decode_options
    )


BACKEND_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = (
    BACKEND_DIR /
    "uploads"
)

LIVE_DATA_DIR = (
    BACKEND_DIR /
    "live_data"
)

LIVE_SESSIONS_DIR = (
    LIVE_DATA_DIR /
    "sessions"
)

DATA_DIR = (
    BACKEND_DIR /
    "data"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LIVE_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LIVE_SESSIONS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HISTORY_FILE = (
    DATA_DIR /
    "history.json"
)


ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm"
}


MAX_FILE_SIZE = 50 * 1024 * 1024


def validate_decoded_payload(
    payload
):

    if not payload:

        raise jwt.InvalidTokenError(
            "JWT payload is empty"
        )

    subject = payload.get(
        "sub"
    )

    if not subject:

        raise jwt.InvalidTokenError(
            "JWT subject is missing"
        )

    role = payload.get(
        "role"
    )

    if role and role != "authenticated":

        raise jwt.InvalidTokenError(
            "JWT role is not authenticated"
        )

    return payload


def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(
        security
    )
):

    if not credentials:

        raise HTTPException(
            status_code=401,
            detail="Missing authentication token"
        )

    try:

        payload = decode_supabase_token(
            credentials.credentials
        )

        return validate_decoded_payload(
            payload
        )

    except jwt.ExpiredSignatureError as error:

        print(
            f"JWT ERROR: Token expired: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )

    except jwt.InvalidAudienceError as error:

        print(
            f"JWT ERROR: Invalid audience: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token audience"
        )

    except jwt.InvalidIssuerError as error:

        print(
            f"JWT ERROR: Invalid issuer: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token issuer"
        )

    except jwt.InvalidSignatureError as error:

        print(
            f"JWT ERROR: Invalid signature: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token signature"
        )

    except jwt.PyJWKClientError as error:

        print(
            f"JWT ERROR: JWKS error: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "JWT signing key could not "
                "be resolved"
            )
        )

    except jwt.InvalidAlgorithmError as error:

        print(
            f"JWT ERROR: Invalid algorithm: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="JWT algorithm is not accepted"
        )

    except jwt.InvalidTokenError as error:

        print(
            f"JWT ERROR: Invalid token: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    except Exception as error:

        print(
            "JWT ERROR: Unexpected authentication "
            f"error: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication verification failed"
            )
        )


def check_rate_limit(
    request: Request
):

    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    now = time.time()

    rate_limit_store[client_ip] = [
        timestamp
        for timestamp in rate_limit_store[client_ip]
        if now - timestamp < RATE_LIMIT_WINDOW
    ]

    if (
        len(rate_limit_store[client_ip])
        >= RATE_LIMIT_REQUESTS
    ):

        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. "
                f"Max {RATE_LIMIT_REQUESTS} "
                f"requests per "
                f"{RATE_LIMIT_WINDOW}s"
            )
        )

    rate_limit_store[client_ip].append(
        now
    )


def authenticate_credentials(
    credentials: HTTPAuthorizationCredentials
):

    if not credentials:

        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    try:

        payload = decode_supabase_token(
            credentials.credentials
        )

        return validate_decoded_payload(
            payload
        )

    except jwt.ExpiredSignatureError as error:

        print(
            f"JWT ERROR: Token expired: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )

    except jwt.InvalidAudienceError as error:

        print(
            f"JWT ERROR: Invalid audience: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token audience"
        )

    except jwt.InvalidIssuerError as error:

        print(
            f"JWT ERROR: Invalid issuer: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token issuer"
        )

    except jwt.InvalidSignatureError as error:

        print(
            f"JWT ERROR: Invalid signature: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token signature"
        )

    except jwt.PyJWKClientError as error:

        print(
            f"JWT ERROR: JWKS error: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "JWT signing key could not "
                "be resolved"
            )
        )

    except jwt.InvalidAlgorithmError as error:

        print(
            f"JWT ERROR: Invalid algorithm: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "JWT algorithm is not accepted"
            )
        )

    except jwt.InvalidTokenError as error:

        print(
            f"JWT ERROR: Invalid token: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    except Exception as error:

        print(
            "JWT ERROR: Unexpected "
            f"authentication error: {error}"
        )

        raise HTTPException(
            status_code=401,
            detail=(
                "Authentication verification "
                "failed"
            )
        )


def require_auth_or_public(
    endpoint_type: str = "protected"
):

    def dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(
            security
        )
    ):

        check_rate_limit(
            request
        )

        if endpoint_type == "protected":

            payload = authenticate_credentials(
                credentials
            )

            request.state.user = payload

            return payload

        if endpoint_type == "optional":

            if credentials:

                try:

                    payload = decode_supabase_token(
                        credentials.credentials
                    )

                    payload = validate_decoded_payload(
                        payload
                    )

                    request.state.user = payload

                    return payload

                except Exception as error:

                    print(
                        f"Optional JWT ignored: {error}"
                    )

            return None

        return None

    return dependency


FFMPEG_PATH_ENV = os.environ.get(
    "FFMPEG_PATH"
)


def get_ffmpeg_path():

    if FFMPEG_PATH_ENV:

        path = Path(
            FFMPEG_PATH_ENV
        )

        if path.exists():

            return path

    if os.name == "nt":

        winget_path = (
            Path.home()
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "ffmpeg-9.0-full_build"
            / "bin"
            / "ffmpeg.exe"
        )

        if winget_path.exists():

            return winget_path

    common_paths = [
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path("/opt/homebrew/bin/ffmpeg"),
        Path("/opt/local/bin/ffmpeg"),
    ]

    for path in common_paths:

        if path.exists():

            return path

    return None


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://deepshield.app",
        "https://www.deepshield.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def current_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


def load_history(
    limit: int = 100
):

    supabase = get_supabase()

    try:
        # Query detection_results directly - it has all core fields including confidence, label
        result = (
            supabase
            .table("detection_results")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        if not result.data:
            return []

        # Get unique audio_record_ids and session_ids for enrichment
        audio_record_ids = list(set(r.get("audio_record_id") for r in result.data if r.get("audio_record_id")))
        session_ids = list(set(r.get("session_id") for r in result.data if r.get("session_id")))

        # Fetch audio_records in bulk
        audio_records_map = {}
        if audio_record_ids:
            ar_result = supabase.table("audio_records").select("*").in_("id", audio_record_ids).execute()
            if ar_result.data:
                audio_records_map = {ar["id"]: ar for ar in ar_result.data}

        # Fetch analysis_sessions in bulk
        sessions_map = {}
        if session_ids:
            sess_result = supabase.table("analysis_sessions").select("*").in_("id", session_ids).execute()
            if sess_result.data:
                sessions_map = {s["id"]: s for s in sess_result.data}

        # Enrich results
        enriched = []
        for row in result.data:
            ar = audio_records_map.get(row.get("audio_record_id"), {})
            sess = sessions_map.get(row.get("session_id"), {})
            enriched.append({
                **row,
                "source": sess.get("source"),
                "filename": ar.get("original_filename"),
                "size_bytes": ar.get("file_size_bytes"),
                "chunk": ar.get("chunk_number"),
                "status": ar.get("status"),
            })
        return enriched

    except Exception as error:
        print(f"History load error: {error}")
        import traceback
        traceback.print_exc()
        return []


def add_history_record(
    record
):

    supabase = get_supabase()

    try:

        session_id = record.get(
            "session_id"
        )

        if session_id:

            session_result = (
                supabase
                .table("analysis_sessions")
                .select("id")
                .eq(
                    "id",
                    session_id
                )
                .execute()
            )

            if not session_result.data:

                session_data = {
                    "id": session_id,
                    "source": record.get(
                        "source",
                        "upload"
                    ),
                    "status": "completed",
                    "started_at": record.get(
                        "created_at"
                    ),
                    "ended_at": record.get(
                        "created_at"
                    ),
                    "total_chunks": (
                        record.get(
                            "chunk",
                            1
                        )
                        if record.get(
                            "source"
                        ) == "extension_live"
                        else 1
                    )
                }

                (
                    supabase
                    .table("analysis_sessions")
                    .insert(
                        session_data
                    )
                    .execute()
                )

        audio_record_data = {
            "session_id": session_id,
            "chunk_number": record.get(
                "chunk"
            ),
            "original_filename": record.get(
                "filename"
            ),
            "storage_path": record.get(
                "storage_path"
            ),
            "storage_bucket": record.get(
                "storage_bucket",
                "audio-files"
            ),
            "file_size_bytes": record.get(
                "size_bytes"
            ),
            "format": record.get(
                "format"
            ),
            "status": "completed",
            "uploaded_at": record.get(
                "created_at"
            ),
            "processed_at": record.get(
                "created_at"
            ),
        }

        audio_result = (
            supabase
            .table("audio_records")
            .insert(
                audio_record_data
            )
            .execute()
        )

        audio_record_id = (
            audio_result.data[0]["id"]
            if audio_result.data
            else None
        )

        db_record = {
            "audio_record_id": audio_record_id,
            "session_id": session_id,
            "label": record["label"],
            "class_id": record["class_id"],
            "confidence": record["confidence"],
            "fake_probability": record.get(
                "fake_probability",
                0
            ),
            "real_probability": record.get(
                "real_probability",
                0
            ),
            "model_version": (
                "RawNet2-PhonemeDF2026"
            ),
            "threshold": 0.5,
            "processing_time_ms": None,
            "created_at": record[
                "created_at"
            ]
        }

        result = (
            supabase
            .table("detection_results")
            .insert(
                db_record
            )
            .execute()
        )

        if result.data:

            return result.data[0]

    except Exception as error:

        print(
            f"Database insert error: {error}"
        )

    return record


def validate_audio_file(
    filename
):

    extension = Path(
        filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    return extension


def find_ffmpeg():

    configured_path = get_ffmpeg_path()

    if configured_path:

        return configured_path

    if os.name == "nt":

        ffmpeg_from_path = subprocess.run(
            [
                "where",
                "ffmpeg"
            ],
            capture_output=True,
            text=True,
            shell=True
        )

    else:

        ffmpeg_from_path = subprocess.run(
            [
                "which",
                "ffmpeg"
            ],
            capture_output=True,
            text=True
        )

    if ffmpeg_from_path.returncode == 0:

        lines = [
            line.strip()
            for line
            in ffmpeg_from_path.stdout.splitlines()
            if line.strip()
        ]

        if lines:

            return Path(
                lines[0]
            )

    raise RuntimeError(
        "FFmpeg was not found. "
        "Set FFMPEG_PATH environment variable "
        "or install FFmpeg in PATH."
    )


def convert_webm_to_wav(
    input_file: Path,
    output_file: Path
):

    ffmpeg = find_ffmpeg()

    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_file)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        error_message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "Unknown FFmpeg error"
        )

        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            + error_message
        )

    if not output_file.exists():

        raise RuntimeError(
            "FFmpeg completed but the WAV file "
            "was not created."
        )

    if output_file.stat().st_size == 0:

        raise RuntimeError(
            "FFmpeg created an empty WAV file."
        )

    return output_file


@app.get("/api/health")
def health_check():

    return {
        "status": "online",
        "service": "DeepShield API"
    }


@app.get("/api/stats")
def get_stats(
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    history = load_history(
        limit=1000
    )

    total = len(history)

    authentic = sum(
        1
        for item in history
        if item.get("label") == "REAL"
    )

    deepfake = sum(
        1
        for item in history
        if item.get("label")
        in {
            "FAKE",
            "DEEPFAKE"
        }
    )

    completed = [
        item
        for item in history
        if item.get("confidence") is not None
    ]

    if completed:
        accuracy = sum(item.get("confidence", 0) for item in completed) / len(completed)
    else:
        accuracy = None

    live_chunks = sum(
        1
        for item in history
        if item.get("source") == "extension_live"
    )

    upload_analyses = sum(
        1
        for item in history
        if item.get("source") == "upload"
    )

    return {
        "total_analyses": total,
        "authentic_audio": authentic,
        "deepfake_detected": deepfake,
        "detection_accuracy": accuracy,
        "live_chunks": live_chunks,
        "upload_analyses": upload_analyses
    }


@app.get("/api/history")
def get_history(
    limit: int = 100,
    request: Request = None,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    history = load_history(
        limit=limit
    )

    return {
        "status": "success",
        "count": len(history),
        "records": history
    }


@app.delete("/api/history")
def clear_history(
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    supabase = get_supabase()

    try:

        (
            supabase
            .table("detection_results")
            .delete()
            .neq(
                "id",
                ""
            )
            .execute()
        )

        return {
            "status": "success",
            "message": "Analysis history cleared"
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to clear history: "
                f"{error}"
            )
        )


@app.post("/api/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    request: Request = None,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    extension = validate_audio_file(
        file.filename
    )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file"
        )

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Audio file exceeds the 50 MB limit"
            )
        )

    original_name = Path(
        file.filename or "audio"
    ).name

    unique_name = (
        f"{uuid4().hex}_"
        f"{original_name}"
    )

    destination = (
        UPLOAD_DIR /
        unique_name
    )

    with destination.open(
        "wb"
    ) as audio_file:

        audio_file.write(
            content
        )

    analysis_id = uuid4().hex

    model_file = destination
    converted_file = None
    storage_info = None

    try:

        if extension == ".webm":

            converted_file = (
                UPLOAD_DIR /
                f"{uuid4().hex}_converted.wav"
            )

            convert_webm_to_wav(
                destination,
                converted_file
            )

            model_file = converted_file

        prediction = predict_audio(
            str(model_file)
        )

        storage_info = upload_audio_file(
            destination,
            folder="uploads"
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("DEEPSHIELD MODEL ERROR")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)
        print()

        cleanup_temp_files(
            [
                destination,
                converted_file
            ]
            if converted_file
            else [
                destination
            ]
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Audio analysis failed: {error}"
            )
        )

    finally:

        cleanup_temp_files(
            [
                destination,
                converted_file
            ]
            if converted_file
            else [
                destination
            ]
        )

    record = {
        "id": analysis_id,
        "source": "upload",
        "filename": original_name,
        "stored_filename": unique_name,
        "size_bytes": len(content),
        "format": extension[1:],
        "status": "COMPLETED",
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "class_id": prediction["class_id"],
        "fake_probability": prediction[
            "fake_probability"
        ],
        "real_probability": prediction[
            "real_probability"
        ],
        "storage_path": (
            storage_info["storage_path"]
            if storage_info
            else None
        ),
        "storage_bucket": (
            storage_info["storage_bucket"]
            if storage_info
            else None
        ),
        "created_at": current_timestamp()
    }

    add_history_record(
        record
    )

    return {
        "status": "success",
        "message": "Audio analyzed successfully",
        "analysis": record,
        "label": prediction["label"],
        "confidence": prediction["confidence"],
        "class_id": prediction["class_id"],
        "fake_probability": prediction[
            "fake_probability"
        ],
        "real_probability": prediction[
            "real_probability"
        ],
    }


@app.post("/api/live-session/start")
def start_live_session(
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    session_id = str(
        uuid4()
    )

    session_dir = (
        LIVE_SESSIONS_DIR /
        session_id
    )

    audio_dir = (
        session_dir /
        "audio"
    )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "session_id": session_id,
        "source": "extension_live",
        "started_at": current_timestamp(),
        "ended_at": None,
        "chunks": 0
    }

    metadata_file = (
        session_dir /
        "metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print(
        "DeepShield live session started: "
        f"{session_id}"
    )

    return {
        "status": "started",
        "session_id": session_id
    }


@app.post("/api/live-analyze")
async def live_analyze(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(None),
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    if not session_id:

        session_id = request.query_params.get(
            "session_id"
        )

    extension = validate_audio_file(
        file.filename
    )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Empty audio chunk"
        )

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Audio chunk exceeds the size limit"
            )
        )

    if not session_id:

        session_id = str(
            uuid4()
        )

        session_dir = (
            LIVE_SESSIONS_DIR /
            session_id
        )

        audio_dir = (
            session_dir /
            "audio"
        )

        audio_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata = {
            "session_id": session_id,
            "source": "extension_live",
            "started_at": current_timestamp(),
            "ended_at": None,
            "chunks": 0
        }

        metadata_file = (
            session_dir /
            "metadata.json"
        )

        with metadata_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2
            )

    session_dir = (
        LIVE_SESSIONS_DIR /
        session_id
    )

    audio_dir = (
        session_dir /
        "audio"
    )

    audio_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        session_dir /
        "metadata.json"
    )

    existing_chunks = list(
        audio_dir.glob(
            "chunk_*"
        )
    )

    chunk_number = (
        len(existing_chunks) + 1
    )

    chunk_filename = (
        f"chunk_{chunk_number:06d}"
        f"{extension}"
    )

    destination = (
        audio_dir /
        chunk_filename
    )

    with destination.open(
        "wb"
    ) as audio_file:

        audio_file.write(
            content
        )

    metadata = {}

    if metadata_file.exists():

        try:

            with metadata_file.open(
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(
                    file
                )

        except Exception:

            metadata = {}

    metadata["session_id"] = session_id

    metadata["source"] = "extension_live"

    metadata["chunks"] = chunk_number

    if "started_at" not in metadata:

        metadata["started_at"] = (
            current_timestamp()
        )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("DEEPSHIELD LIVE AUDIO")
    print("=" * 70)
    print(
        f"Session: {session_id}"
    )
    print(
        f"Chunk: {chunk_number}"
    )
    print(
        f"File: {destination}"
    )
    print(
        f"Size: {len(content)} bytes"
    )
    print(
        f"Format: {extension}"
    )
    print("=" * 70)

    converted_file = None
    model_file = destination
    storage_info = None

    try:

        if extension == ".webm":

            converted_file = (
                audio_dir /
                f"chunk_{chunk_number:06d}_model.wav"
            )

            print(
                "Converting WebM to WAV..."
            )

            convert_webm_to_wav(
                destination,
                converted_file
            )

            model_file = converted_file

            print(
                f"Converted file: {converted_file}"
            )

        prediction = predict_audio(
            str(model_file)
        )

        print(
            "MODEL RESULT:",
            prediction
        )

        storage_info = upload_audio_file(
            destination,
            folder=(
                f"live_sessions/{session_id}"
            )
        )

        record = {
            "id": uuid4().hex,
            "source": "extension_live",
            "session_id": session_id,
            "filename": chunk_filename,
            "stored_filename": str(
                Path("live_data")
                / "sessions"
                / session_id
                / "audio"
                / chunk_filename
            ),
            "size_bytes": len(content),
            "format": extension[1:],
            "status": "COMPLETED",
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "class_id": prediction["class_id"],
            "fake_probability": prediction[
                "fake_probability"
            ],
            "real_probability": prediction[
                "real_probability"
            ],
            "chunk": chunk_number,
            "storage_path": (
                storage_info["storage_path"]
                if storage_info
                else None
            ),
            "storage_bucket": (
                storage_info["storage_bucket"]
                if storage_info
                else None
            ),
            "created_at": current_timestamp()
        }

        add_history_record(
            record
        )

        print(
            "DEEPSHIELD LIVE ANALYSIS SUCCESS"
        )

        print(
            f"Label: {prediction['label']}"
        )

        print(
            "Confidence: "
            f"{prediction['confidence']}%"
        )

        print("=" * 70)
        print()

        return {
            "status": "audio_received",
            "session_id": session_id,
            "chunk": chunk_number,
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "class_id": prediction["class_id"],
            "fake_probability": prediction[
                "fake_probability"
            ],
            "real_probability": prediction[
                "real_probability"
            ],
            "state": "COMPLETED",
            "message": (
                "Live audio chunk analyzed "
                "successfully."
            )
        }

    except Exception as error:

        print()
        print("=" * 70)
        print("DEEPSHIELD LIVE MODEL ERROR")
        print("=" * 70)

        print(
            f"Session: {session_id}"
        )

        print(
            f"Chunk: {chunk_number}"
        )

        print(
            f"Audio file: {destination}"
        )

        print(
            f"Error: {error}"
        )

        traceback.print_exc()

        print("=" * 70)
        print()

        record = {
            "id": uuid4().hex,
            "source": "extension_live",
            "session_id": session_id,
            "filename": chunk_filename,
            "stored_filename": str(
                Path("live_data")
                / "sessions"
                / session_id
                / "audio"
                / chunk_filename
            ),
            "size_bytes": len(content),
            "format": extension[1:],
            "status": "ERROR",
            "label": None,
            "confidence": None,
            "class_id": None,
            "error": str(error),
            "created_at": current_timestamp()
        }

        add_history_record(
            record
        )

        return {
            "status": "audio_received",
            "session_id": session_id,
            "chunk": chunk_number,
            "label": None,
            "confidence": None,
            "class_id": None,
            "state": "ERROR",
            "error": str(error),
            "message": (
                "Live audio was captured, "
                "but model analysis failed."
            )
        }

    finally:

        cleanup_temp_files(
            [
                destination,
                converted_file
            ]
            if converted_file
            else [
                destination
            ]
        )


@app.post("/api/live-session/stop")
def stop_live_session(
    session_id: str,
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    session_dir = (
        LIVE_SESSIONS_DIR /
        session_id
    )

    if not session_dir.exists():

        raise HTTPException(
            status_code=404,
            detail="Live session not found"
        )

    metadata_file = (
        session_dir /
        "metadata.json"
    )

    metadata = {}

    if metadata_file.exists():

        try:

            with metadata_file.open(
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(
                    file
                )

        except Exception:

            metadata = {}

    metadata["session_id"] = session_id

    metadata["ended_at"] = (
        current_timestamp()
    )

    with metadata_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    import shutil

    shutil.rmtree(
        session_dir,
        ignore_errors=True
    )

    return {
        "status": "stopped",
        "session_id": session_id,
        "chunks": metadata.get(
            "chunks",
            0
        )
    }


@app.get("/api/dataset")
def get_dataset(
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):

    history = load_history()

    upload_records = [
        item
        for item in history
        if item.get(
            "source"
        ) == "upload"
    ]

    live_records = [
        item
        for item in history
        if item.get(
            "source"
        ) == "extension_live"
    ]

    sessions = set(
        item.get(
            "session_id"
        )
        for item in live_records
        if item.get(
            "session_id"
        )
    )

    total_size = sum(
        item.get(
            "size_bytes",
            0
        )
        for item in history
    )

    return {
        "status": "success",
        "total_files": len(history),
        "upload_files": len(upload_records),
        "live_chunks": len(live_records),
        "live_sessions": len(sessions),
        "total_size_bytes": total_size
    }


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_split_stats(split_dir: Path) -> dict:
    import pandas as pd

    stats = {}
    for split_name in ["train", "validation", "test"]:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            stats[split_name] = {"real": 0, "fake": 0, "total": 0, "speakers": 0, "generators": []}
            continue
        df = pd.read_csv(csv_path)
        real_count = int((df["target"] == 1).sum())
        fake_count = int((df["target"] == 0).sum())
        speakers = df["speaker"].nunique() if "speaker" in df.columns else 0
        generators = df["generator"].unique().tolist() if "generator" in df.columns else []
        stats[split_name] = {
            "real": real_count,
            "fake": fake_count,
            "total": int(len(df)),
            "speakers": int(speakers),
            "generators": generators,
        }
    return stats


def compute_dfadd_stats(dfadd_root: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"train": {"real": 0, "fake": 0, "total": 0}, "valid": {"real": 0, "fake": 0, "total": 0}, "test": {"real": 0, "fake": 0, "total": 0}}

    stats = {}
    for split_name in ["train", "valid", "test"]:
        files = list(dfadd_root.glob(f"{split_name}*.parquet"))
        total_fake = 0
        total_real = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["label", "split"], batch_size=10000):
                df = batch.to_pandas()
                if split_name == "train":
                    split_mask = df["split"] == "train"
                elif split_name == "valid":
                    split_mask = df["split"] == "valid"
                else:
                    split_mask = df["split"] == "test"
                split_df = df[split_mask]
                for label in split_df["label"]:
                    mapped = str(label).strip().lower()
                    if mapped in {"spoofed", "spoof", "fake", "synthetic", "0"}:
                        total_fake += 1
                    elif mapped in {"real", "bonafide", "genuine", "1"}:
                        total_real += 1
        stats[split_name] = {
            "real": total_real,
            "fake": total_fake,
            "total": total_real + total_fake,
        }
    return stats


@app.get("/api/training-dataset")
def get_training_dataset(
    request: Request,
    _: None = Depends(
        require_auth_or_public(
            "protected"
        )
    )
):
    splits_root = PROJECT_ROOT / "ml" / "data" / "splits" / "PhonemeDF_2026_v2"
    phonemedf_clean_root = PROJECT_ROOT / "ml" / "data" / "splits" / "PhonemeDF_2026_clean"
    dfadd_root = PROJECT_ROOT / "ml" / "data" / "external" / "DFADD" / "data"
    root_splits = PROJECT_ROOT / "ml" / "data" / "splits"

    datasets = {}

    # PhonemeDF 2026 v2 (speaker/generator disjoint)
    if splits_root.exists():
        speaker_disjoint = compute_split_stats(splits_root / "speaker_disjoint")
        generator_disjoint = compute_split_stats(splits_root / "generator_disjoint")
        datasets["PhonemeDF_2026_v2"] = {
            "description": "Speaker and generator disjoint splits for robust deepfake audio detection",
            "splits": {
                "speaker_disjoint": speaker_disjoint,
                "generator_disjoint": generator_disjoint,
            },
        }

    # PhonemeDF 2026 Clean (speaker/generator disjoint)
    if phonemedf_clean_root.exists():
        clean_speaker = compute_split_stats(phonemedf_clean_root / "speaker_disjoint") if (phonemedf_clean_root / "speaker_disjoint").exists() else compute_split_stats(phonemedf_clean_root)
        clean_generator = compute_split_stats(phonemedf_clean_root / "generator_disjoint") if (phonemedf_clean_root / "generator_disjoint").exists() else {}
        datasets["PhonemeDF_2026_clean"] = {
            "description": "Clean version of PhonemeDF 2026 with disjoint splits",
            "splits": {
                "speaker_disjoint": clean_speaker,
                "generator_disjoint": clean_generator,
            },
        }

    # Root splits (ASVspoof 2019 combined)
    if (root_splits / "train.csv").exists():
        root_stats = compute_split_stats(root_splits)
        datasets["ASVspoof_2019_Combined"] = {
            "description": "ASVspoof 2019 LA combined with other datasets (legacy combined splits)",
            "splits": root_stats,
        }

    # DFADD (DeepFake Audio Detection Dataset)
    if dfadd_root.exists():
        dfadd_stats = compute_dfadd_stats(dfadd_root)
        datasets["DFADD"] = {
            "description": "DeepFake Audio Detection Dataset - large-scale multi-generator TTS/VC dataset",
            "splits": dfadd_stats,
        }

    # Compute grand totals
    grand_total = 0
    grand_real = 0
    grand_fake = 0
    for ds in datasets.values():
        for split_stats in ds["splits"].values():
            grand_total += split_stats.get("total", 0)
            grand_real += split_stats.get("real", 0)
            grand_fake += split_stats.get("fake", 0)

    return {
        "status": "success",
        "datasets": datasets,
        "summary": {
            "total_samples": grand_total,
            "total_real": grand_real,
            "total_fake": grand_fake,
            "label_mapping": {"0": "FAKE (spoof)", "1": "REAL (bonafide)"},
            "datasets_count": len(datasets),
        },
    }