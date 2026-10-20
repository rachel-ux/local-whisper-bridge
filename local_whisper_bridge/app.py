from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse

from .engine import LocalWhisperEngine, repair_model, selected_model
from .lab import compare

MAX_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg", ".webm"}


def create_app(engine: Any | None = None) -> FastAPI:
    app = FastAPI(title="Local Whisper Bridge", version="0.1.0")
    transcription_engine = engine if engine is not None else LocalWhisperEngine()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok", "model": selected_model(), "model_ready": transcription_engine.ready(),
            "repair_model": repair_model(), "repair_model_ready": transcription_engine.repair_ready(),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": selected_model(), "object": "model", "owned_by": "local"}]}

    @app.get("/lab")
    def lab_page():
        return FileResponse(Path(__file__).with_name("lab.html"))

    @app.post("/lab/compare")
    async def lab_compare(
        file: UploadFile = File(...),
        language: str | None = Form(None),
        reference: str | None = Form(None),
        authorization: str | None = Header(None),
    ):
        token = os.environ.get("LWB_TOKEN")
        if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "A valid local bearer token is required")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, "Upload WAV, MP3, M4A, MP4, FLAC, OGG, or WebM audio")
        if not transcription_engine.ready():
            raise HTTPException(503, "Model is not prepared. Run the prepare command first.")
        path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="lwb-lab-", suffix=suffix, delete=False) as output:
                path = output.name
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Audio file exceeds the 25 MB limit")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, "Audio file is empty")
            try:
                return await run_in_threadpool(compare, path, language, reference)
            except (RuntimeError, TimeoutError) as exc:
                raise HTTPException(422, str(exc)) from exc
        finally:
            await file.close()
            if path:
                Path(path).unlink(missing_ok=True)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        strategy: str = Form("fast"),
        authorization: str | None = Header(None),
    ):
        token = os.environ.get("LWB_TOKEN")
        if token and not secrets.compare_digest(authorization or "", f"Bearer {token}"):
            raise HTTPException(401, "A valid local bearer token is required")
        if model != selected_model():
            raise HTTPException(400, f"This server has only the {selected_model()!r} model")
        if response_format not in {"json", "text", "verbose_json"}:
            raise HTTPException(400, "Supported response_format values: json, text, verbose_json")
        if strategy not in {"fast", "adaptive"}:
            raise HTTPException(400, "Supported strategy values: fast, adaptive")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"Supported audio extensions: {', '.join(sorted(ALLOWED_SUFFIXES))}")
        if not transcription_engine.ready():
            raise HTTPException(503, "Model is not prepared. Run: local-whisper-bridge prepare")
        if strategy == "adaptive" and not transcription_engine.repair_ready():
            raise HTTPException(503, "Repair model is not prepared. Run: local-whisper-bridge prepare --model base")

        path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix="lwb-", suffix=suffix, delete=False) as output:
                path = output.name
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "Audio file exceeds the 25 MB limit")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, "Audio file is empty")
            try:
                result = await run_in_threadpool(transcription_engine.transcribe, path, language, prompt, strategy)
            except Exception as exc:
                raise HTTPException(422, f"Unable to decode or transcribe audio: {exc}") from exc
            if response_format == "text":
                return PlainTextResponse(result["text"])
            if response_format == "json":
                return {"text": result["text"]}
            return {"task": "transcribe", **result}
        finally:
            await file.close()
            if path:
                Path(path).unlink(missing_ok=True)

    return app


app = create_app()
