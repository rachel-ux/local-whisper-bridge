from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse

from .local_engine import LocalWhisperEngine, prepared
from .openai_whisper import OpenAIWhisperClient, OpenAIWhisperError, api_key

MAX_BYTES = 25 * 1024 * 1024
OPENAI_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm"}


def create_app(cloud: Any | None = None, offline: Any | None = None) -> FastAPI:
    app = FastAPI(title="OpenAI Whisper Call Tester", version="0.1.0")
    cloud_client = cloud if cloud is not None else OpenAIWhisperClient()
    offline_engine = offline if offline is not None else LocalWhisperEngine()

    @app.get("/")
    def home():
        return RedirectResponse("/openai")

    @app.get("/openai")
    def openai_page():
        return FileResponse(Path(__file__).with_name("openai.html"))

    @app.get("/openai/status")
    def openai_status():
        return {"configured": bool(api_key()), "offline_ready": prepared(), "model": "whisper-1"}

    @app.post("/offline/transcribe")
    async def offline_transcribe(request: Request, file: UploadFile = File(...), rescue: bool = True):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin transcription requests are blocked")
        if not prepared() and offline is None:
            raise HTTPException(503, "Offline model is missing. Run: local-whisper-bridge prepare")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in OPENAI_SUFFIXES:
            raise HTTPException(400, "Upload WAV, MP3, M4A, MP4, MPEG, MPGA, or WebM audio")
        path = None
        try:
            with tempfile.NamedTemporaryFile(prefix="whisper-offline-", suffix=suffix, delete=False) as output:
                path = Path(output.name)
                content = await file.read(MAX_BYTES + 1)
                if len(content) > MAX_BYTES:
                    raise HTTPException(413, "Audio file exceeds the 25 MB limit")
                if not content:
                    raise HTTPException(400, "Audio file is empty")
                output.write(content)
            try:
                return await run_in_threadpool(offline_engine.transcribe, str(path), rescue)
            except (RuntimeError, ValueError) as exc:
                raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            if path is not None:
                path.unlink(missing_ok=True)

    @app.post("/openai/transcribe")
    async def openai_transcribe(request: Request, file: UploadFile = File(...)):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Cross-origin transcription requests are blocked")
        if not api_key():
            raise HTTPException(503, "Add OPENAI_API_KEY to .env.local or the server environment first.")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in OPENAI_SUFFIXES:
            raise HTTPException(400, "OpenAI Whisper accepts WAV, MP3, M4A, MP4, MPEG, MPGA, or WebM")
        try:
            content = await file.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise HTTPException(413, "Audio file exceeds the 25 MB limit")
            if not content:
                raise HTTPException(400, "Audio file is empty")
            try:
                return await cloud_client.transcribe(content, file.filename or "audio.webm")
            except OpenAIWhisperError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc
        finally:
            await file.close()

    return app


app = create_app()
