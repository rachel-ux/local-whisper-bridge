"""Same-checkpoint CPU comparison for the optimization proof of concept."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from .engine import model_path, selected_model
from .precision import allocate, rank_candidates


def _words(value: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", value.casefold(), re.UNICODE)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    expected, actual = _words(reference), _words(hypothesis)
    if not expected:
        return None
    previous = list(range(len(actual) + 1))
    for index, word in enumerate(expected, 1):
        current = [index]
        for j, candidate in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (word != candidate),
            ))
        previous = current
    return round(previous[-1] / len(expected), 3)


def run_variant(
    audio_path: str, name: str, compute_type: str, language: str | None,
    start: float | None = None, end: float | None = None,
) -> dict:
    command = [
        sys.executable, "-m", "local_whisper_bridge.lab_worker", audio_path,
        "--model", name, "--compute-type", compute_type,
    ]
    if language:
        command.extend(["--language", language])
    if start is not None and end is not None:
        command.extend(["--start", str(start), "--end", str(end)])
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=240, check=False,
        env=os.environ.copy(),
    )
    if completed.returncode:
        # Avoid reflecting arbitrary audio paths or environment values into the UI.
        raise RuntimeError(f"{compute_type} inference failed (exit {completed.returncode}).")
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{compute_type} inference returned invalid metrics.") from exc


def _speech_spans(audio_path: str) -> list[dict]:
    os.environ.setdefault("ORT_DISABLE_ALL_TELEMETRY", "1")
    import onnxruntime

    onnxruntime.disable_telemetry_events()
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(audio_path, sampling_rate=16000)
    spans = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=350))
    return [{"start": span["start"] / 16000, "end": span["end"] / 16000} for span in spans]


def _adaptive(audio_path: str, name: str, draft: dict, language: str | None, budget_percent: int) -> dict:
    begin = time.perf_counter()
    speech = _speech_spans(audio_path)
    budget = draft["audio_seconds"] * budget_percent / 100
    ranked = rank_candidates(draft["segments"], speech, draft["audio_seconds"])
    chosen = allocate(ranked, budget)
    updated = [dict(item) for item in draft["segments"]]
    decisions = []
    peak = draft["peak_process_mb"]
    extra_inference = 0.0
    for candidate in chosen:
        repair = run_variant(audio_path, name, "float32", language or draft["language"], candidate.start, candidate.end)
        if candidate.segment_index is not None:
            original = updated[candidate.segment_index]
            selected_parts = [
                part["text"] for part in repair["segments"]
                if float(original["start"]) <= candidate.start + (part["start"] + part["end"]) / 2 <= float(original["end"])
            ]
        else:
            selected_parts = [
                part["text"] for part in repair["segments"]
                if not any(
                    float(item["start"]) <= candidate.start + (part["start"] + part["end"]) / 2 <= float(item["end"])
                    for item in updated
                )
            ]
        replacement = " ".join(selected_parts).strip()
        peak = max(peak, repair["peak_process_mb"])
        extra_inference += repair["inference_seconds"]
        before = updated[candidate.segment_index]["text"] if candidate.segment_index is not None else ""
        if replacement:
            if candidate.segment_index is not None:
                updated[candidate.segment_index]["text"] = replacement
            else:
                updated.append({"start": candidate.start, "end": candidate.end, "text": replacement})
        decisions.append({
            "start": round(candidate.start, 2), "end": round(candidate.end, 2),
            "risk": candidate.risk, "reason": candidate.reason,
            "before": before, "after": replacement,
            "changed": bool(replacement and replacement != before),
        })
    updated.sort(key=lambda item: item["start"])
    elapsed = time.perf_counter() - begin
    return {
        "compute_type": "int8 + selective float32",
        "model": name,
        "text": " ".join(item["text"] for item in updated).strip(),
        "language": draft["language"],
        "audio_seconds": draft["audio_seconds"],
        "load_seconds": draft["load_seconds"],
        "inference_seconds": round(draft["inference_seconds"] + extra_inference, 3),
        "total_seconds": round(draft["total_seconds"] + elapsed, 3),
        "peak_process_mb": peak,
        "repair_audio_seconds": round(sum(item.end - item.start for item in chosen), 3),
        "budget_seconds": round(budget, 3),
        "budget_overrun_seconds": round(max(0.0, sum(item.end - item.start for item in chosen) - budget), 3),
        "candidates_found": len(ranked),
        "decisions": decisions,
    }


def compare(
    audio_path: str, language: str | None = None, reference: str | None = None,
    budget_percent: int = 30,
) -> dict:
    name = selected_model()
    if not 0 <= budget_percent <= 100:
        raise ValueError("Budget percent must be between 0 and 100")
    if not (model_path(name) / "model.bin").is_file():
        raise RuntimeError(f"The {name} model is not prepared.")
    baseline = run_variant(audio_path, name, "float32", language)
    optimized = run_variant(audio_path, name, "int8", language)
    adaptive = _adaptive(audio_path, name, optimized, language, budget_percent)
    if reference and reference.strip():
        baseline["word_error_rate"] = word_error_rate(reference, baseline["text"])
        optimized["word_error_rate"] = word_error_rate(reference, optimized["text"])
        adaptive["word_error_rate"] = word_error_rate(reference, adaptive["text"])
    else:
        baseline["word_error_rate"] = None
        optimized["word_error_rate"] = None
        adaptive["word_error_rate"] = None
    return {
        "checkpoint": name,
        "model_file_mb": round((model_path(name) / "model.bin").stat().st_size / (1024 * 1024), 1),
        "device": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor()},
        "baseline": baseline,
        "optimized": optimized,
        "adaptive": adaptive,
        "method": "Same Whisper checkpoint: full CPU float32, full CPU int8, then int8 plus risk-directed float32 rechecks within the chosen audio budget.",
        "limitations": "The risk score is a heuristic, not calibrated confidence; float32 may be equally wrong. Peak memory is the highest individual process, including Python overhead. Sequential runs can benefit differently from caches. No energy or Apple Watch measurement.",
    }
