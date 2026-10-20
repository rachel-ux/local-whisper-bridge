from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from local_whisper_bridge.app import create_app
from local_whisper_bridge.local_engine import (
    LocalWhisperEngine, choose_gaps, plan_quiet_gaps,
)


def sample_audio() -> np.ndarray:
    audio = np.full(10 * 16000, 0.0002, dtype=np.float32)
    audio[16000:32000] = 0.2
    audio[64000:80000] = 0.006
    return audio


def test_quiet_gap_planner_skips_transcribed_speech_and_respects_budget():
    gaps = plan_quiet_gaps(sample_audio(), [(1.0, 2.0)])
    chosen = choose_gaps(gaps, 10.0)
    assert len(chosen) == 1
    assert chosen[0].start < 4.0 < chosen[0].end
    assert chosen[0].end > 5.0
    assert chosen[0].end - chosen[0].start <= 2.0


def test_recovery_pass_uses_actual_second_decode(monkeypatch):
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda *_args, **_kwargs: sample_audio())

    class Model:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **kwargs):
            self.calls.append(kwargs)
            if kwargs["vad_filter"]:
                return iter([SimpleNamespace(start=1.0, end=2.0, text="Primary")]), SimpleNamespace(duration_after_vad=1.0)
            return iter([SimpleNamespace(start=0.1, end=0.7, text="quiet words", avg_logprob=-0.2, no_speech_prob=0.1)]), None

    model = Model()
    result = LocalWhisperEngine(model).transcribe("ignored.wav")
    assert result["text"] == "Primary quiet words"
    assert result["optimization"]["recovered_segments"] == 1
    assert len(result["optimization"]["retried_gaps"]) == 1
    assert len(model.calls) == 2
    assert model.calls[1]["vad_filter"] is False


def test_offline_route_is_separate_from_openai(monkeypatch):
    class Engine:
        def transcribe(self, path, rescue):
            assert path.endswith(".wav")
            assert rescue is False
            return {"text": "local result", "provider": "local"}

    client = TestClient(create_app(offline=Engine()))
    response = client.post(
        "/offline/transcribe?rescue=false",
        files={"file": ("voice.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "local"
