"""Offline faster-whisper speech transcription."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseModelPipeline, PipelineRegistry

@PipelineRegistry.register("whisper")
class WhisperPipeline(BaseModelPipeline):
    """Transcribe a local audio file using a CTranslate2 Whisper checkpoint."""

    input_kind: ClassVar[str] = "file"
    backend: ClassVar[str] = "ctranslate2"

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        super().__init__(model_path, device)
        self._model: Any | None = None

    def load_model(self) -> None:
        """Load a pre-downloaded checkpoint without Hugging Face downloads."""
        if not self.model_path.is_dir() or not (self.model_path / "model.bin").is_file():
            raise FileNotFoundError(f"Whisper model is missing: {self.model_path}")
        from faster_whisper import WhisperModel
        threads = int(os.environ.get("WHISPER_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
        if threads < 1:
            raise ValueError("WHISPER_CPU_THREADS must be positive")
        self._model = WhisperModel(
            str(self.model_path),
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=threads if self.device == "cpu" else 0,
            num_workers=1,
            local_files_only=True,
        )

    def run(self, input_data: Any) -> dict[str, Any]:
        """Return language, duration, and timestamped speech segments."""
        if self._model is None:
            raise RuntimeError("Whisper model has not been loaded")
        path = Path(input_data)
        if not path.is_file():
            raise ValueError("audio file does not exist")
        segments, info = self._model.transcribe(
            str(path), beam_size=1, vad_filter=False, temperature=0.0
        )
        # faster-whisper decodes lazily, so consume the generator before the
        # request's temporary file is removed.
        items = [
            {"start": round(segment.start, 3), "end": round(segment.end, 3), "text": segment.text.strip()}
            for segment in segments
            if segment.text.strip()
        ]
        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_seconds": round(info.duration, 3),
            "segments": items,
            "text": " ".join(item["text"] for item in items),
            "device": self.device,
            "compute_type": self.compute_type,
        }

