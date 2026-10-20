"""Real, server-side OpenAI Whisper transcription calls."""

from __future__ import annotations

import os
from pathlib import Path

import httpx

OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL = "whisper-1"


def api_key() -> str | None:
    """Read a key from the process or an ignored local env file; never log it."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(os.environ.get("LWB_ENV_FILE", Path.cwd() / ".env.local"))
    if not env_path.is_file():
        return None
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                return value.strip().strip('"\'') or None
    except OSError:
        return None
    return None


class OpenAIWhisperError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class OpenAIWhisperClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self.transport = transport

    async def transcribe(
        self, content: bytes, filename: str, language: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        key = api_key()
        if not key:
            raise OpenAIWhisperError(503, "Add OPENAI_API_KEY to .env.local or the server environment first.")
        data = {"model": MODEL, "response_format": "json"}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        suffix = Path(filename).suffix.lower()
        media_type = {
            ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
            ".mp4": "audio/mp4", ".webm": "audio/webm", ".mpeg": "audio/mpeg",
            ".mpga": "audio/mpeg",
        }.get(suffix, "application/octet-stream")
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=120.0) as client:
                response = await client.post(
                    OPENAI_TRANSCRIPTIONS_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files={"file": (Path(filename).name, content, media_type)},
                )
        except httpx.TimeoutException as exc:
            raise OpenAIWhisperError(504, "OpenAI transcription timed out.") from exc
        except httpx.RequestError as exc:
            raise OpenAIWhisperError(502, "Could not reach OpenAI. Check the connection.") from exc
        if response.status_code == 401:
            raise OpenAIWhisperError(401, "OpenAI rejected the API key.")
        if response.status_code == 429:
            raise OpenAIWhisperError(429, "OpenAI rate limit or quota reached.")
        if not response.is_success:
            raise OpenAIWhisperError(502, f"OpenAI transcription failed (HTTP {response.status_code}).")
        try:
            result = response.json()
        except ValueError as exc:
            raise OpenAIWhisperError(502, "OpenAI returned invalid JSON.") from exc
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise OpenAIWhisperError(502, "OpenAI returned no transcript text.")
        return {"text": result["text"], "model": MODEL, "provider": "openai"}
