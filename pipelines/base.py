"""Shared contracts and device selection for locally stored models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, TypeVar


def resolve_device_and_compute(
    requested_device: str = "auto", backend: str = "torch"
) -> tuple[str, str]:
    """Select a supported device and precision for the model's inference runtime.

    Automatic selection prefers CUDA, MPS, then CPU. CTranslate2 has no MPS
    backend, so Whisper skips it and uses CPU when CUDA is unavailable.
    """
    requested = requested_device.lower().strip()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if backend not in {"torch", "ctranslate2"}:
        raise ValueError(f"unknown inference backend: {backend}")
    if backend == "ctranslate2" and requested == "mps":
        raise ValueError("faster-whisper cannot run on MPS; use device=cpu or auto")

    def supported_precision(device: str) -> str | None:
        """Return the usable precision, or None when this runtime cannot use it."""
        precision = "int8" if device == "cpu" else "float16"
        if backend == "torch":
            return precision
        import ctranslate2

        try:
            supported = ctranslate2.get_supported_compute_types(device)
        except (RuntimeError, ValueError):
            return None
        if precision in supported:
            return precision
        return "float32" if device == "cpu" and "float32" in supported else None

    if requested == "cpu":
        precision = supported_precision("cpu")
        if precision is None:
            raise ValueError(f"{backend} cannot run on CPU")
        return "cpu", precision

    try:
        import torch
    except ImportError:
        if requested != "auto":
            raise ValueError(f"{requested} requires PyTorch and an available device") from None
        precision = supported_precision("cpu")
        if precision is None:
            raise ValueError(f"{backend} cannot run on CPU")
        return "cpu", precision

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    candidates = [requested] if requested != "auto" else ["cuda", "mps", "cpu"]
    for device in candidates:
        if device == "cuda" and not cuda_available:
            continue
        if device == "mps" and (not mps_available or backend == "ctranslate2"):
            continue
        precision = supported_precision(device)
        if precision is not None:
            return device, precision
    if requested == "auto":
        raise ValueError(f"{backend} has no usable inference device")
    raise ValueError(f"{backend} cannot run on requested device: {requested}")


class BaseModelPipeline(ABC):
    """Contract for a model that loads once and accepts one inference input."""

    input_kind: ClassVar[str]
    backend: ClassVar[str] = "torch"

    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.requested_device = device.lower().strip()
        self.device, self.compute_type = resolve_device_and_compute(device, self.backend)

    @abstractmethod
    def load_model(self) -> None:
        """Load weights strictly from ``model_path`` without network access."""

    @abstractmethod
    def run(self, input_data: Any) -> dict[str, Any]:
        """Run inference and return a JSON-serializable result."""


PipelineT = TypeVar("PipelineT", bound=type[BaseModelPipeline])


class PipelineRegistry:
    """Decorator-based registry of pipeline implementations."""

    _pipelines: ClassVar[dict[str, type[BaseModelPipeline]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[PipelineT], PipelineT]:
        """Register a unique pipeline name at import time."""
        if not name or not name.strip():
            raise ValueError("pipeline name cannot be empty")

        def decorator(pipeline_class: PipelineT) -> PipelineT:
            if not issubclass(pipeline_class, BaseModelPipeline):
                raise TypeError("registered pipelines must inherit BaseModelPipeline")
            if name in cls._pipelines:
                raise ValueError(f"pipeline already registered: {name}")
            cls._pipelines[name] = pipeline_class
            return pipeline_class

        return decorator

    @classmethod
    def create(
        cls, name: str, model_path: str | Path, device: str = "auto"
    ) -> BaseModelPipeline:
        """Instantiate a registered pipeline; loading remains explicit."""
        try:
            pipeline_class = cls._pipelines[name]
        except KeyError as exc:
            raise KeyError(f"unknown pipeline: {name}") from exc
        return pipeline_class(model_path=model_path, device=device)

