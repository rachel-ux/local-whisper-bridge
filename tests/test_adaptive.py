from local_whisper_bridge.adaptive import plan_coverage_gaps, plan_repairs
from local_whisper_bridge.engine import LocalWhisperEngine
from types import SimpleNamespace
import numpy as np


def segment(start, end, logprob):
    return {"start": start, "end": end, "text": "words", "avg_logprob": logprob,
            "compression_ratio": 1.2, "no_speech_prob": 0.01, "temperature": 0.0}


def test_only_uncertain_spans_are_selected():
    spans = [segment(0, 8, -0.2), segment(8, 16, -0.9), segment(16, 24, -0.3)]
    choices = plan_repairs(spans, 24)
    assert [choice.index for choice in choices] == [1]
    assert choices[0].reason == "low_logprob"


def test_budget_prioritizes_risk_per_second():
    spans = [segment(0, 8, -1.0), segment(8, 16, -0.6), segment(16, 24, -0.2)]
    choices = plan_repairs(spans, 24, minimum_budget_seconds=8)
    assert [choice.index for choice in choices] == [0]


def test_no_risky_segments_spend_no_secondary_compute():
    assert plan_repairs([segment(0, 10, -0.1)], 10) == []


def test_spoken_transcript_hole_gets_budget_priority():
    transcript = [segment(0, 11, -0.1), segment(24, 35, -0.1)]
    speech = [{"start": 13.0, "end": 21.5}]
    gaps = plan_coverage_gaps(transcript, speech, 35, 15)
    assert len(gaps) == 1
    assert gaps[0].start == 11
    assert gaps[0].end == 24


def test_engine_repairs_only_the_selected_span(monkeypatch):
    def result(text, start, end, logprob):
        return SimpleNamespace(
            text=text, start=start, end=end, avg_logprob=logprob,
            compression_ratio=1.0, no_speech_prob=0.0, temperature=0.0,
        )

    class DraftModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, *_args, **_kwargs):
            self.calls += 1
            item = result("clear", 0, 10, -0.1) if self.calls == 1 else result("uncertain", 0, 10, -0.9)
            return iter([item]), SimpleNamespace(language="en", duration=12.0)

    class RepairModel:
        def __init__(self):
            self.lengths = []

        def transcribe(self, audio, **_kwargs):
            self.lengths.append(len(audio))
            return iter([result("corrected", 0, 10, -0.1)]), None

    repair = RepairModel()
    draft = DraftModel()
    engine = LocalWhisperEngine()
    monkeypatch.setattr(engine, "_get_model", lambda name: draft if name == "tiny" else repair)
    monkeypatch.setattr("faster_whisper.audio.decode_audio", lambda *_args, **_kwargs: np.zeros(24 * 16000, dtype=np.float32))
    answer = engine.transcribe("unused.wav", None, None, "adaptive")

    assert answer["text"] == "clear corrected"
    assert answer["optimization"]["repair_seconds"] == 10.0
    assert answer["optimization"]["repaired_segments"] == 1
    assert repair.lengths == [10 * 16000]
