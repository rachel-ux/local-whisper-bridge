"""Contract tests for the offline registry and API lifecycle."""

from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient

from main import create_app
from pipelines.base import BaseModelPipeline, PipelineRegistry, resolve_device_and_compute
from pipelines.whisper_pipeline import WhisperPipeline


@PipelineRegistry.register("test-file")
class FilePipeline(BaseModelPipeline):
    input_kind = "file"
    last_path: Path | None = None
    load_count = 0

    def load_model(self) -> None:
        type(self).load_count += 1

    def run(self, input_data):
        type(self).last_path = Path(input_data)
        return {"size": type(self).last_path.stat().st_size}


@PipelineRegistry.register("test-text")
class TextPipeline(BaseModelPipeline):
    input_kind = "text"

    def load_model(self) -> None:
        pass

    def run(self, input_data):
        return {"upper": input_data.upper()}


@PipelineRegistry.register("test-failure")
class FailurePipeline(BaseModelPipeline):
    input_kind = "file"
    last_path: Path | None = None

    def load_model(self) -> None:
        pass

    def run(self, input_data):
        type(self).last_path = Path(input_data)
        raise RuntimeError("internal failure")


def test_registry_device_resolution_and_unique_names():
    assert resolve_device_and_compute("cpu") == ("cpu", "int8")
    assert isinstance(PipelineRegistry.create("test-text", ".", "cpu"), TextPipeline)
    try:
        resolve_device_and_compute("invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid device was accepted")


def test_device_precedence_and_whisper_mps_fallback(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device_and_compute("auto") == ("cuda", "float16")
    fake_torch.cuda.is_available = lambda: False
    assert resolve_device_and_compute("auto") == ("mps", "float16")
    whisper = WhisperPipeline("models/whisper-base", "auto")
    assert (whisper.device, whisper.compute_type) == ("cpu", "int8")
    try:
        WhisperPipeline("models/whisper-base", "mps")
    except ValueError:
        pass
    else:
        raise AssertionError("explicit unsupported MPS was accepted")


def test_whisper_resolver_checks_runtime_gpu_support(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    fake_ctranslate2 = SimpleNamespace(
        get_supported_compute_types=lambda device: (
            {"int8", "float32"} if device == "cpu" else set()
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)

    assert resolve_device_and_compute("auto", "ctranslate2") == ("cpu", "int8")
    assert (WhisperPipeline("models/whisper-base").device) == "cpu"
    try:
        resolve_device_and_compute("cuda", "ctranslate2")
    except ValueError:
        pass
    else:
        raise AssertionError("explicit unsupported CUDA was accepted")


def test_models_load_once_and_temporary_audio_is_deleted():
    FilePipeline.load_count = 0
    config = {
        "audio": {"pipeline": "test-file", "model_path": ".", "device": "cpu"},
        "text": {"pipeline": "test-text", "model_path": ".", "device": "cpu"},
    }
    with TestClient(create_app(config)) as client:
        assert set(client.get("/health").json()["models"]) == {"audio", "text"}
        response = client.post(
            "/api/v1/infer/audio", files={"file": ("voice.wav", b"abc", "audio/wav")}
        )
        assert response.status_code == 200
        assert response.json()["result"]["size"] == 3
        assert FilePipeline.last_path is not None
        assert not FilePipeline.last_path.exists()
        assert client.post("/api/v1/infer/text", data={"text": "hello"}).json()["result"] == {"upper": "HELLO"}
        assert client.post("/api/v1/infer/missing", data={"text": "x"}).status_code == 404
        assert client.post("/api/v1/infer/audio", data={"text": "x"}).status_code == 400
        assert client.post("/api/v1/infer/text", data={"text": " "}).status_code == 400
    assert FilePipeline.load_count == 1


def test_rejected_upload_still_cleans_temporary_file():
    config = {"audio": {"pipeline": "test-file", "model_path": ".", "device": "cpu"}}
    with TestClient(create_app(config)) as client:
        assert client.post(
            "/api/v1/infer/audio", files={"file": ("voice.wav", b"", "audio/wav")}
        ).status_code == 400


def test_processing_failure_returns_500_and_removes_audio():
    config = {"broken": {"pipeline": "test-failure", "model_path": ".", "device": "cpu"}}
    with TestClient(create_app(config)) as client:
        response = client.post(
            "/api/v1/infer/broken", files={"file": ("voice.wav", b"abc", "audio/wav")}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "model inference failed"
        assert FailurePipeline.last_path is not None
        assert not FailurePipeline.last_path.exists()
