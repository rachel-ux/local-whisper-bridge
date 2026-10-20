from fastapi.testclient import TestClient

from local_whisper_bridge.app import create_app


class FakeEngine:
    def __init__(self):
        self.calls = []

    def ready(self):
        return True

    def repair_ready(self):
        return True

    def transcribe(self, path, language, prompt, strategy):
        with open(path, "rb") as audio:
            payload = audio.read()
        self.calls.append((payload, language, prompt, strategy))
        return {
            "text": "Hello world",
            "language": "en",
            "duration": 1.25,
            "segments": [{"id": 0, "start": 0.0, "end": 1.25, "text": "Hello world"}],
        }


def test_upload_reaches_engine_and_returns_openai_style_text():
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "tiny", "language": "en", "prompt": "context"},
        files={"file": ("sample.wav", b"RIFFaudio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": "Hello world"}
    assert engine.calls == [(b"RIFFaudio", "en", "context", "fast")]


def test_verbose_response_and_rejection_of_wrong_model():
    client = TestClient(create_app(FakeEngine()))
    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "tiny", "response_format": "verbose_json"},
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )
    assert response.json()["segments"][0]["text"] == "Hello world"
    wrong = client.post(
        "/v1/audio/transcriptions",
        data={"model": "canto"},
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )
    assert wrong.status_code == 400


def test_token_and_empty_upload(monkeypatch):
    monkeypatch.setenv("LWB_TOKEN", "secret")
    client = TestClient(create_app(FakeEngine()))
    data = {"model": "tiny"}
    files = {"file": ("sample.wav", b"audio", "audio/wav")}
    assert client.post("/v1/audio/transcriptions", data=data, files=files).status_code == 401
    response = client.post(
        "/v1/audio/transcriptions", data=data,
        files={"file": ("sample.wav", b"", "audio/wav")},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400
