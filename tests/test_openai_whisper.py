from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from local_whisper_bridge.app import create_app
from local_whisper_bridge.openai_whisper import OpenAIWhisperClient


class ReadyEngine:
    def ready(self):
        return True


def test_real_openai_multipart_path_and_response(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer test-only-key"
        assert b'name="model"' in request.content
        assert b"whisper-1" in request.content
        assert b'name="language"' in request.content
        assert b"hello-audio" in request.content
        return httpx.Response(200, json={"text": "Real response shape."})

    app = create_app(ReadyEngine(), OpenAIWhisperClient(httpx.MockTransport(handler)))
    client = TestClient(app)
    assert client.get("/openai").status_code == 200
    assert client.get("/openai/status").json()["configured"] is True
    response = client.post(
        "/openai/transcribe",
        files={"file": ("voice.webm", b"hello-audio", "audio/webm")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Real response shape.", "model": "whisper-1", "provider": "openai"}
    assert len(observed) == 1


def test_missing_key_and_cross_origin_request(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LWB_ENV_FILE", "/private/tmp/nonexistent-openai-key.env")
    client = TestClient(create_app(ReadyEngine()))
    assert client.get("/openai/status").json()["configured"] is False
    upload = {"file": ("voice.webm", b"audio", "audio/webm")}
    assert client.post("/openai/transcribe", files=upload).status_code == 503
    response = client.post(
        "/openai/transcribe", files=upload,
        headers={"Origin": "https://unrelated.example"},
    )
    assert response.status_code == 403


def test_openai_key_rejection_is_safe(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {"message": "secret or personal text"}}))
    client = TestClient(create_app(ReadyEngine(), OpenAIWhisperClient(transport)))
    response = client.post("/openai/transcribe", files={"file": ("voice.wav", b"audio", "audio/wav")})
    assert response.status_code == 401
    assert "secret or personal text" not in json.dumps(response.json())
