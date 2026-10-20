"""Same-checkpoint CPU comparison for the optimization proof of concept."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from .engine import model_path, selected_model


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


def run_variant(audio_path: str, name: str, compute_type: str, language: str | None) -> dict:
    command = [
        sys.executable, "-m", "local_whisper_bridge.lab_worker", audio_path,
        "--model", name, "--compute-type", compute_type,
    ]
    if language:
        command.extend(["--language", language])
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


def compare(audio_path: str, language: str | None = None, reference: str | None = None) -> dict:
    name = selected_model()
    if not (model_path(name) / "model.bin").is_file():
        raise RuntimeError(f"The {name} model is not prepared.")
    baseline = run_variant(audio_path, name, "float32", language)
    optimized = run_variant(audio_path, name, "int8", language)
    if reference and reference.strip():
        baseline["word_error_rate"] = word_error_rate(reference, baseline["text"])
        optimized["word_error_rate"] = word_error_rate(reference, optimized["text"])
    else:
        baseline["word_error_rate"] = None
        optimized["word_error_rate"] = None
    return {
        "checkpoint": name,
        "model_file_mb": round((model_path(name) / "model.bin").stat().st_size / (1024 * 1024), 1),
        "device": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor()},
        "baseline": baseline,
        "optimized": optimized,
        "method": "CTranslate2 CPU float32 versus CPU int8, same weights and decode settings; separate processes",
        "limitations": "Peak process memory includes Python and runtime overhead. Sequential runs may benefit differently from filesystem caches. int8 can change output quality. This is not an energy or Apple Watch measurement.",
    }
