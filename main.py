"""FastAPI server for strictly offline, registered model pipelines."""

from __future__ import annotations

import os

# Set thread and offline policy before importing any ML runtime. Override the
# vCPU default with WHISPER_CPU_THREADS after benchmarking on target hardware.
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))
if CPU_THREADS < 1:
    raise ValueError("WHISPER_CPU_THREADS / OMP_NUM_THREADS must be positive")
os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["WHISPER_CPU_THREADS"] = str(CPU_THREADS)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, TypedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

import pipelines  # noqa: F401: registers the built-in implementations
from pipelines.base import BaseModelPipeline, PipelineRegistry

LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
AUDIO_VIDEO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}


class ModelConfig(TypedDict):
    """One named, locally stored model loaded during application startup."""

    pipeline: str
    model_path: str | Path
    device: str


def configured_models() -> dict[str, ModelConfig]:
    """Build the model configuration without fetching weights."""
    config: dict[str, ModelConfig] = {
        "whisper": {
            "pipeline": "whisper",
            "model_path": os.environ.get(
                "WHISPER_MODEL_PATH", str(PROJECT_ROOT / "models" / "whisper-base")
            ),
            "device": os.environ.get("WHISPER_DEVICE", "auto"),
        }
    }
    text_path = os.environ.get("TEXT_MODEL_PATH")
    if text_path:
        config["text-classifier"] = {
            "pipeline": "text-classifier",
            "model_path": text_path,
            "device": os.environ.get("TEXT_MODEL_DEVICE", "auto"),
        }
    return config


def create_app(config: dict[str, ModelConfig] | None = None) -> FastAPI:
    """Create an app whose configured models are loaded exactly once."""
    model_config = config if config is not None else configured_models()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded: dict[str, BaseModelPipeline] = {}
        for name, settings in model_config.items():
            pipeline = PipelineRegistry.create(
                settings["pipeline"], settings["model_path"], settings["device"]
            )
            await run_in_threadpool(pipeline.load_model)
            loaded[name] = pipeline
            LOG.info("Loaded %s from %s on %s", name, pipeline.model_path, pipeline.device)
        app.state.pipelines = loaded
        # A single request per model prevents CPU thread-pool oversubscription.
        app.state.locks = {name: asyncio.Semaphore(1) for name in loaded}
        try:
            yield
        finally:
            app.state.pipelines = {}
            app.state.locks = {}

    app = FastAPI(title="Offline Model Inference", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """List loaded pipelines and their actual execution devices."""
        return {
            "status": "ready",
            "offline": True,
            "models": {
                name: {"pipeline": type(pipeline).__name__, "device": pipeline.device,
                       "compute_type": pipeline.compute_type}
                for name, pipeline in app.state.pipelines.items()
            },
        }

    @app.post("/api/v1/infer/{model_name}")
    async def infer(
        model_name: str,
        file: UploadFile | None = File(default=None),
        text: str | None = Form(default=None),
    ) -> dict[str, Any]:
        """Infer from one uploaded audio/video file or one form text field."""
        pipeline: BaseModelPipeline | None = app.state.pipelines.get(model_name)
        if pipeline is None:
            raise HTTPException(status_code=404, detail=f"model not loaded: {model_name}")
        if (file is None) == (text is None):
            raise HTTPException(status_code=400, detail="provide exactly one of file or text")
        if pipeline.input_kind == "file" and file is None:
            raise HTTPException(status_code=400, detail="this model requires a file")
        if pipeline.input_kind == "text" and text is None:
            raise HTTPException(status_code=400, detail="this model requires text")

        temp_path: Path | None = None
        try:
            if file is not None:
                suffix = Path(file.filename or "").suffix.lower()
                if suffix not in AUDIO_VIDEO_SUFFIXES:
                    raise HTTPException(status_code=400, detail="unsupported audio/video extension")
                with tempfile.NamedTemporaryFile(prefix="offline-infer-", suffix=suffix, delete=False) as target:
                    temp_path = Path(target.name)
                    size = 0
                    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                        size += len(chunk)
                        if size > MAX_UPLOAD_BYTES:
                            raise HTTPException(status_code=400, detail="file exceeds 25 MiB")
                        target.write(chunk)
                if size == 0:
                    raise HTTPException(status_code=400, detail="file is empty")
                input_data: Any = temp_path
            else:
                if not text or not text.strip():
                    raise HTTPException(status_code=400, detail="text must be nonempty")
                input_data = text

            async with app.state.locks[model_name]:
                result = await run_in_threadpool(pipeline.run, input_data)
            return {"model": model_name, "result": result}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOG.exception("Inference failed for %s", model_name)
            raise HTTPException(status_code=500, detail="model inference failed") from exc
        finally:
            if file is not None:
                await file.close()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    return app


app = create_app()

