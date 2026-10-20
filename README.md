# Offline Model Inference API

A local-only FastAPI service with a registry of model pipelines. The default model is a pre-downloaded Whisper Base checkpoint running through `faster-whisper`. A text classifier can be enabled by pointing `TEXT_MODEL_PATH` at a locally stored Hugging Face sequence-classification checkpoint. The inference server never downloads weights or calls a hosted AI API.

```text
local-whisper-bridge/
├── pipelines/
│   ├── __init__.py
│   ├── base.py                 # Device resolver, abstract contract, registry
│   ├── whisper_pipeline.py     # Offline speech recognition
│   └── text_pipeline.py        # Offline text classification
├── models/
│   └── whisper-base/           # Prepared local checkpoint, ignored by Git
├── main.py                     # Thread settings, model lifecycle, API
└── requirements.txt
```

## Run

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8765
```

The prepared `models/whisper-base` checkpoint is available on this development computer. It is ignored by Git, so on a new machine **prepare the weights once while online, before starting the offline server**:

```bash
python - <<'PY'
from faster_whisper.utils import download_model
download_model("base", output_dir="models/whisper-base")
PY
```

After preparation, disconnect the network if desired and start the server. Startup fails clearly if `model.bin` is missing. The endpoint will not substitute a remote model.

The service binds to loopback in the example command. Do not expose it to other hosts without adding authentication, request limits, and deployment-specific controls.

## Inference

The generic endpoint accepts multipart form data. Use a `file` for Whisper or `text` for a text classifier; send exactly one. It loads each configured model once during startup, queues concurrent requests per model, and deletes uploaded temporary files even when inference fails.

```bash
curl -sS -F "file=@recording.wav" http://127.0.0.1:8765/api/v1/infer/whisper
curl -sS http://127.0.0.1:8765/health
```

The Whisper response includes detected language, language probability, duration, timestamped segments, joined text, and the actual device and compute type. Uploads are limited to 25 MiB. Supported input extensions are WAV, MP3, M4A, MP4, MPEG, MPGA, WebM, OGG, and FLAC; `faster-whisper` decodes the file locally.

To enable text classification, first prepare a compatible sequence-classification model and tokenizer into a local directory, then set `TEXT_MODEL_PATH` before startup. This development computer has a DistilBERT sentiment checkpoint in `models/text-sentiment`, but model files are ignored by Git. On another machine, prepare it while online:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    local_dir="models/text-sentiment",
    allow_patterns=["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"],
)
PY
```

The server will only read these files. Its offline policy remains active when the text pipeline is enabled.

```bash
TEXT_MODEL_PATH="$PWD/models/text-sentiment" uvicorn main:app --host 127.0.0.1 --port 8765
curl -sS -F "text=This is useful" http://127.0.0.1:8765/api/v1/infer/text-classifier
```

`GET /health` lists loaded models. The inference endpoint returns `404` for an unloaded name, `400` for invalid input, and a generic `500` for an inference failure; detailed failures remain in server logs.

## How hardware selection works

`resolve_device_and_compute("auto", backend=...)` centrally checks CUDA, then MPS, then CPU against the selected runtime's capabilities. It chooses `float16` for a supported GPU or `int8` for CPU (falling back to CPU `float32` only if that runtime lacks int8). Pipelines declare their inference backend instead of choosing their own device. Set `WHISPER_DEVICE` or `TEXT_MODEL_DEVICE` to `cpu`, `cuda`, or `mps` to request a specific device; unavailable explicit devices fail at startup.

There is an important backend distinction: **CTranslate2/faster-whisper does not support MPS**. On an Apple Silicon Mac without CUDA, the resolver selects **CPU int8 for Whisper** and **MPS float16 for the PyTorch classifier**. Explicit `WHISPER_DEVICE=mps` fails with an explanation. If PyTorch sees CUDA but the installed CTranslate2 build cannot use it, automatic Whisper selection falls back to CPU; an explicit unsupported request fails. On CPU the text classifier attempts dynamic int8 quantization of supported linear layers. If its PyTorch build lacks a required quantized backend, it uses float32 and reports that actual precision in `/health` and inference responses.

`main.py` sets `OMP_NUM_THREADS` from the available logical CPUs **before importing either ML runtime**. `WHISPER_CPU_THREADS` overrides this default and is also passed to CTranslate2. More threads are not always faster: benchmark the actual model and audio length on your hardware, especially when several requests arrive together.

```bash
WHISPER_CPU_THREADS=4 uvicorn main:app --host 127.0.0.1 --port 8765
```

The code forces Hugging Face and Transformers offline mode and passes `local_files_only=True` when loading both pipeline types. These settings prevent model fetches; they do not make a model small or guarantee real-time transcription on every device. For an isolated deployment, install dependencies and prepare model files beforehand, then run in an environment with outbound access disabled.

## Add another model

Subclass `BaseModelPipeline`, set `input_kind` to `"file"` or `"text"`, implement `load_model()` and `run(input_data)`, and decorate the class with `@PipelineRegistry.register("your-name")`. Import its module from `pipelines/__init__.py`, then add a local path to the configuration returned by `configured_models()` in `main.py`. Keep model loading local and report the device and precision actually used.

The default Whisper path uses `beam_size=1` for CPU throughput and does not apply VAD, which avoids automatically dropping very quiet speech. This is a deliberate baseline, not a universal accuracy optimum. Benchmark latency, transcription error, memory, and power before changing beam size or CPU thread count for production.
