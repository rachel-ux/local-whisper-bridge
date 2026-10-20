from __future__ import annotations

from fastapi.testclient import TestClient

from local_whisper_bridge.app import create_app
from local_whisper_bridge.lab import word_error_rate


def test_word_error_rate_handles_code_switching_and_punctuation() -> None:
    assert word_error_rate("नमस्ते, Rachel!", "नमस्ते Rachel") == 0.0
    assert word_error_rate("hello world", "hello") == 0.5
    assert word_error_rate("", "hello") is None


def test_lab_page_and_compare_route(monkeypatch) -> None:
    class ReadyEngine:
        def ready(self):
            return True

    def fake_compare(path, language, reference):
        assert language == "en"
        assert reference == "hello"
        with open(path, "rb") as audio:
            assert audio.read() == b"RIFFsample"
        return {"checkpoint": "tiny", "baseline": {"text": "hello"}, "optimized": {"text": "hello"}}

    monkeypatch.setattr("local_whisper_bridge.app.compare", fake_compare)
    client = TestClient(create_app(ReadyEngine()))
    assert client.get("/lab").status_code == 200
    response = client.post(
        "/lab/compare",
        files={"file": ("clip.wav", b"RIFFsample", "audio/wav")},
        data={"language": "en", "reference": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["baseline"]["text"] == "hello"
