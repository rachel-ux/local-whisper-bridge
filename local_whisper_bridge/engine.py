from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from .adaptive import plan_coverage_gaps, plan_repairs, repair_budget

SUPPORTED_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "large-v3")
SAMPLE_RATE = 16000


def model_root() -> Path:
    return Path(os.environ.get("LWB_MODEL_DIR", Path.home() / ".cache" / "local-whisper-bridge")).expanduser()


def selected_model() -> str:
    name = os.environ.get("LWB_MODEL", "tiny")
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}. Choose one of: {', '.join(SUPPORTED_MODELS)}")
    return name


def repair_model() -> str:
    name = os.environ.get("LWB_REPAIR_MODEL", "base")
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported repair model {name!r}")
    return name


def model_path(name: str) -> Path:
    if name not in SUPPORTED_MODELS:
        raise ValueError("Unsupported model")
    return model_root() / name


def _segments_to_result(segments: Any, info: Any, model_name: str) -> dict[str, Any]:
    items = [
        {
            "id": i,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "avg_logprob": segment.avg_logprob,
            "compression_ratio": segment.compression_ratio,
            "no_speech_prob": segment.no_speech_prob,
            "temperature": segment.temperature,
            "source_model": model_name,
        }
        for i, segment in enumerate(segments)
    ]
    return {
        "text": " ".join(item["text"] for item in items).strip(),
        "language": info.language,
        "duration": info.duration,
        "segments": items,
    }


class LocalWhisperEngine:
    """Local two-stage transcription. Models are loaded lazily and held in RAM."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = Lock()

    def ready(self) -> bool:
        return (model_path(selected_model()) / "model.bin").is_file()

    def repair_ready(self) -> bool:
        return repair_model() != selected_model() and (model_path(repair_model()) / "model.bin").is_file()

    def _get_model(self, name: str) -> Any:
        if name not in self._models:
            from faster_whisper import WhisperModel

            self._models[name] = WhisperModel(
                str(model_path(name)),
                device=os.environ.get("LWB_DEVICE", "cpu"),
                compute_type=os.environ.get("LWB_COMPUTE_TYPE", "int8"),
                local_files_only=True,
            )
        return self._models[name]

    def transcribe(
        self, audio_path: str, language: str | None, prompt: str | None, strategy: str = "fast"
    ) -> dict[str, Any]:
        # Serialize model work so simultaneous uploads do not overload a CPU.
        with self._lock:
            name = selected_model()
            audio = None
            if strategy == "adaptive":
                from faster_whisper.audio import decode_audio

                audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
                items: list[dict[str, Any]] = []
                detected_language = language
                window_samples = 12 * SAMPLE_RATE
                for start in range(0, len(audio), window_samples):
                    end = min(len(audio), start + window_samples)
                    window_segments, info = self._get_model(name).transcribe(
                        audio[start:end], language=detected_language,
                        initial_prompt=prompt, beam_size=5, temperature=0.0,
                    )
                    detected_language = detected_language or info.language
                    for item in _segments_to_result(window_segments, info, name)["segments"]:
                        item["start"] = max(start / SAMPLE_RATE, item["start"] + start / SAMPLE_RATE)
                        item["end"] = min(end / SAMPLE_RATE, item["end"] + start / SAMPLE_RATE)
                        item["id"] = len(items)
                        items.append(item)
                draft = {
                    "text": " ".join(item["text"] for item in items).strip(),
                    "language": detected_language,
                    "duration": len(audio) / SAMPLE_RATE,
                    "segments": items,
                }
            else:
                fast_segments, fast_info = self._get_model(name).transcribe(
                    audio_path, language=language, initial_prompt=prompt, beam_size=5, temperature=0.0
                )
                draft = _segments_to_result(fast_segments, fast_info, name)
            draft["optimization"] = {
                "strategy": strategy,
                "fast_model": name,
                "repair_model": None,
                "repair_seconds": 0.0,
                "repair_fraction": 0.0,
                "repaired_segments": 0,
                "recovered_gaps": 0,
            }
            if strategy != "adaptive":
                return draft
            assert audio is not None
            budget = repair_budget(draft["duration"])
            gaps = []
            if draft["duration"] >= 20.0:
                from faster_whisper.vad import VadOptions, get_speech_timestamps

                speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500))
                speech_seconds = [
                    {"start": span["start"] / SAMPLE_RATE, "end": span["end"] / SAMPLE_RATE}
                    for span in speech
                ]
                gaps = plan_coverage_gaps(draft["segments"], speech_seconds, draft["duration"], budget)
            gap_seconds = sum(gap.end - gap.start for gap in gaps)
            choices = plan_repairs(
                draft["segments"], draft["duration"], budget_seconds=max(0.0, budget - gap_seconds)
            )
            if not gaps and not choices:
                return draft

            bigger_name = repair_model()
            bigger = self._get_model(bigger_name)
            used_seconds = 0.0
            repaired = 0
            recovered_gaps = 0
            inserted: list[dict[str, Any]] = []
            for gap in gaps:
                start = max(0, int(gap.start * SAMPLE_RATE))
                end = min(len(audio), int(gap.end * SAMPLE_RATE))
                if end <= start:
                    continue
                refined_segments, _ = bigger.transcribe(
                    audio[start:end],
                    language=language or draft["language"],
                    initial_prompt=prompt,
                    beam_size=5,
                    temperature=0.0,
                )
                recovered = 0
                for segment in refined_segments:
                    if segment.text.strip():
                        inserted.append({
                            "id": 0,
                            "start": gap.start + segment.start,
                            "end": gap.start + segment.end,
                            "text": segment.text.strip(),
                            "avg_logprob": segment.avg_logprob,
                            "compression_ratio": segment.compression_ratio,
                            "no_speech_prob": segment.no_speech_prob,
                            "temperature": segment.temperature,
                            "source_model": bigger_name,
                            "repair_reason": "speech_without_transcript",
                        })
                        recovered += 1
                used_seconds += (end - start) / SAMPLE_RATE
                if recovered:
                    recovered_gaps += 1
            for choice in choices:
                item = draft["segments"][choice.index]
                start = max(0, int(float(item["start"]) * SAMPLE_RATE))
                end = min(len(audio), int(float(item["end"]) * SAMPLE_RATE))
                if end <= start:
                    continue
                refined_segments, _ = bigger.transcribe(
                    audio[start:end],
                    language=language or draft["language"],
                    initial_prompt=prompt,
                    beam_size=5,
                    temperature=0.0,
                )
                refined_parts = list(refined_segments)
                refined_text = " ".join(segment.text.strip() for segment in refined_parts).strip()
                used_seconds += (end - start) / SAMPLE_RATE
                if refined_text:
                    item["text"] = refined_text
                    item["source_model"] = bigger_name
                    item["repair_reason"] = choice.reason
                    item["draft_avg_logprob"] = item["avg_logprob"]
                    item["avg_logprob"] = (
                        sum(segment.avg_logprob for segment in refined_parts) / len(refined_parts)
                    )
                    repaired += 1
            draft["segments"].extend(inserted)
            draft["segments"].sort(key=lambda item: (item["start"], item["end"]))
            for i, item in enumerate(draft["segments"]):
                item["id"] = i
            draft["text"] = " ".join(item["text"] for item in draft["segments"]).strip()
            draft["optimization"].update(
                repair_model=bigger_name,
                repair_seconds=round(used_seconds, 3),
                repair_fraction=round(used_seconds / draft["duration"], 3) if draft["duration"] else 0.0,
                repaired_segments=repaired,
                recovered_gaps=recovered_gaps,
            )
            return draft
