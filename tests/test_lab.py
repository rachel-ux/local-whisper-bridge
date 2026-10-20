from __future__ import annotations

from fastapi.testclient import TestClient

from local_whisper_bridge.app import create_app
from local_whisper_bridge.lab import word_error_rate
from local_whisper_bridge.lab import _adaptive
from local_whisper_bridge.precision import allocate, rank_candidates


def test_word_error_rate_handles_code_switching_and_punctuation() -> None:
    assert word_error_rate("नमस्ते, Rachel!", "नमस्ते Rachel") == 0.0
    assert word_error_rate("hello world", "hello") == 0.5
    assert word_error_rate("", "hello") is None


def test_lab_page_and_compare_route(monkeypatch) -> None:
    class ReadyEngine:
        def ready(self):
            return True

    def fake_compare(path, language, reference, budget_percent):
        assert language == "en"
        assert reference == "hello"
        assert budget_percent == 30
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


def test_precision_controller_ranks_evidence_and_enforces_budget() -> None:
    segments = [
        {"start": 0.5, "end": 2.0, "text": "clear", "avg_logprob": -0.08, "compression_ratio": 1.1, "no_speech_prob": 0.02},
        {"start": 3.0, "end": 4.5, "text": "maybe", "avg_logprob": -1.1, "compression_ratio": 1.2, "no_speech_prob": 0.1},
    ]
    speech = [{"start": 0.5, "end": 2.0}, {"start": 3.0, "end": 4.5}, {"start": 7.0, "end": 8.0}]
    candidates = rank_candidates(segments, speech, 10.0)
    assert any(item.reason == "speech without transcript" for item in candidates)
    assert any(item.segment_index == 1 for item in candidates)
    assert all(item.segment_index != 0 for item in candidates)
    selected = allocate(candidates, budget_seconds=2.5)
    assert sum(item.end - item.start for item in selected) <= 2.5
    assert allocate(candidates, budget_seconds=0) == []
    assert allocate(candidates, budget_seconds=0.1), "one risky utterance may exceed a nonzero target"


def test_adaptive_rechecks_with_same_checkpoint_and_records_change(monkeypatch) -> None:
    monkeypatch.setattr("local_whisper_bridge.lab._speech_spans", lambda path: [])

    def fake_variant(path, name, compute_type, language, start, end):
        assert name == "tiny"
        assert compute_type == "float32"
        assert start < 2 < end
        return {
            "text": "corrected words",
            "segments": [{"start": 0.4, "end": 1.6, "text": "corrected words"}],
            "peak_process_mb": 250,
            "inference_seconds": 0.7,
        }

    monkeypatch.setattr("local_whisper_bridge.lab.run_variant", fake_variant)
    draft = {
        "model": "tiny", "text": "wrong words", "language": "en", "audio_seconds": 5.0,
        "load_seconds": 0.2, "inference_seconds": 0.5, "total_seconds": 0.7,
        "peak_process_mb": 200,
        "segments": [{"start": 2.0, "end": 3.5, "text": "wrong words", "avg_logprob": -1.2,
                      "compression_ratio": 1.2, "no_speech_prob": 0.1}],
    }
    result = _adaptive("unused.wav", "tiny", draft, "en", budget_percent=30)
    assert result["text"] == "corrected words"
    assert result["decisions"][0]["changed"] is True
    assert result["decisions"][0]["reason"] == "low decoder probability"
