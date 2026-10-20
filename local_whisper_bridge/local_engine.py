"""Offline Whisper inference with a bounded quiet-gap recovery pass."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_SECONDS = 0.1
MAX_REPAIR_FRACTION = 0.2
MAX_REPAIR_SECONDS = 30.0


def model_dir() -> Path:
    return Path(os.environ.get("LWB_MODEL_DIR", Path.cwd() / "models")).expanduser() / "tiny"


def prepared() -> bool:
    return (model_dir() / "model.bin").is_file()


@dataclass(frozen=True)
class Gap:
    start: float
    end: float
    score: float


def plan_quiet_gaps(audio: np.ndarray, covered_spans: list[tuple[float, float]]) -> list[Gap]:
    """Rank above-floor, low-level regions with little transcript coverage."""
    frame_size = int(SAMPLE_RATE * FRAME_SECONDS)
    if len(audio) < frame_size:
        return []
    rms = np.array([
        float(np.sqrt(np.mean(audio[i:i + frame_size] ** 2)))
        for i in range(0, len(audio) - frame_size + 1, frame_size)
    ])
    floor = max(float(np.percentile(rms, 20)), 0.0003)
    lower = max(floor * 2.5, 0.0015)
    upper = max(floor * 30, 0.08)
    selected = (rms >= lower) & (rms <= upper)
    # A two-frame bridge keeps a softly spoken syllable from fragmenting.
    for i in range(1, len(selected) - 1):
        if selected[i - 1] and selected[i + 1]:
            selected[i] = True
    gaps: list[Gap] = []
    start: int | None = None
    for i in range(len(selected) + 1):
        active = i < len(selected) and bool(selected[i])
        if active and start is None:
            start = i
        if not active and start is not None:
            if (i - start) * FRAME_SECONDS >= 0.3:
                begin = max(0.0, start * FRAME_SECONDS - 0.25)
                end = min(len(audio) / SAMPLE_RATE, i * FRAME_SECONDS + 0.25)
                overlap = sum(max(0.0, min(end, b) - max(begin, a)) for a, b in covered_spans)
                if overlap / (end - begin) < 0.15:
                    level = float(np.mean(rms[start:i]))
                    # Favor sustained speech-like energy above the estimated floor.
                    gaps.append(Gap(begin, end, (end - begin) * min(level / floor, 12)))
            start = None
    return sorted(gaps, key=lambda gap: gap.score, reverse=True)


def choose_gaps(gaps: list[Gap], audio_seconds: float) -> list[Gap]:
    budget = min(MAX_REPAIR_SECONDS, audio_seconds * MAX_REPAIR_FRACTION)
    chosen: list[Gap] = []
    used = 0.0
    for gap in gaps:
        if used + gap.end - gap.start > budget:
            continue
        if any(max(gap.start, other.start) < min(gap.end, other.end) for other in chosen):
            continue
        chosen.append(gap)
        used += gap.end - gap.start
    return sorted(chosen, key=lambda gap: gap.start)


class LocalWhisperEngine:
    def __init__(self, model=None):
        self._model = model

    def _get_model(self):
        if self._model is None:
            if not prepared():
                raise RuntimeError("Offline model is missing. Run: local-whisper-bridge prepare")
            import ctranslate2
            from faster_whisper import WhisperModel

            supported = ctranslate2.get_supported_compute_types("cpu")
            compute_type = "int8" if "int8" in supported else "float32"
            self._model = WhisperModel(str(model_dir()), device="cpu", compute_type=compute_type)
        return self._model

    def transcribe(self, path: str, rescue: bool = True) -> dict:
        from faster_whisper.audio import decode_audio

        started = time.perf_counter()
        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        if not duration:
            raise ValueError("Audio contains no samples")
        model = self._get_model()
        segments, info = model.transcribe(audio, beam_size=1, vad_filter=True)
        primary = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments if s.text.strip()
        ]
        elapsed_primary = time.perf_counter() - started
        retried: list[dict] = []
        recovered: list[dict] = []
        if rescue:
            spans = [(item["start"], item["end"]) for item in primary]
            for gap in choose_gaps(plan_quiet_gaps(audio, spans), duration):
                clip = audio[int(gap.start * SAMPLE_RATE):int(gap.end * SAMPLE_RATE)]
                candidates, _ = model.transcribe(
                    clip, beam_size=3, temperature=0.0, vad_filter=False,
                    condition_on_previous_text=False,
                )
                accepted = []
                for item in candidates:
                    if item.text.strip() and item.avg_logprob > -0.75 and item.no_speech_prob < 0.5:
                        accepted.append({
                            "start": round(gap.start + item.start, 2),
                            "end": round(gap.start + item.end, 2),
                            "text": item.text.strip(),
                        })
                retried.append({"start": round(gap.start, 2), "end": round(gap.end, 2), "recovered": bool(accepted)})
                recovered.extend(accepted)
        all_segments = sorted(primary + recovered, key=lambda item: item["start"])
        elapsed = time.perf_counter() - started
        return {
            "text": " ".join(item["text"] for item in all_segments),
            "segments": all_segments,
            "provider": "local",
            "model": "tiny",
            "optimization": {
                "name": "quiet-gap-rescue" if rescue else "vad-int8-baseline",
                "audio_seconds": round(duration, 2),
                "vad_kept_seconds": round(info.duration_after_vad, 2),
                "primary_seconds": round(elapsed_primary, 2),
                "total_seconds": round(elapsed, 2),
                "real_time_factor": round(elapsed / duration, 3),
                "retried_gaps": retried,
                "recovered_segments": len(recovered),
            },
        }
