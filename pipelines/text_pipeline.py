"""Offline Hugging Face text classification from a local checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from .base import BaseModelPipeline, PipelineRegistry

LOG = logging.getLogger(__name__)


@PipelineRegistry.register("text-classifier")
class TextClassifierPipeline(BaseModelPipeline):
    """Classify text with locally stored model and tokenizer files."""

    input_kind: ClassVar[str] = "text"

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        super().__init__(model_path, device)
        self._classifier: Any | None = None

    def load_model(self) -> None:
        """Load and prepare a classifier without consulting the model hub."""
        if not self.model_path.is_dir() or not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"Text model is missing: {self.model_path}")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        )
        if self.device == "cpu":
            # Dynamic quantization targets Linear layers; other operations
            # remain at normal precision. Some PyTorch builds (including this
            # macOS build) have no quantized::linear_prepack implementation.
            engines = [
                engine for engine in torch.backends.quantized.supported_engines
                if engine != "none"
            ]
            if engines:
                torch.backends.quantized.engine = next(
                    (engine for engine in ("x86", "fbgemm", "qnnpack") if engine in engines),
                    engines[0],
                )
                try:
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except RuntimeError as exc:
                    LOG.warning("CPU int8 text quantization unavailable: %s", exc)
                    self.compute_type = "float32"
            else:
                LOG.warning("CPU int8 text quantization unavailable; using float32")
                self.compute_type = "float32"
        else:
            model = model.to(device=self.device, dtype=torch.float16)
        model.eval()
        self._classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer,
            device=torch.device(self.device),
        )

    def run(self, input_data: Any) -> dict[str, Any]:
        """Return label and score for one nonempty text string."""
        if self._classifier is None:
            raise RuntimeError("Text classifier has not been loaded")
        if not isinstance(input_data, str) or not input_data.strip():
            raise ValueError("text must be a nonempty string")
        predictions = self._classifier(input_data, truncation=True)
        return {
            "predictions": [
                {"label": str(item["label"]), "score": float(item["score"])}
                for item in predictions
            ],
            "device": self.device,
            "compute_type": self.compute_type,
        }

