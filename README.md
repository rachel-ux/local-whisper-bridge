# Local Whisper Bridge

A small FastAPI service that lets another app send an audio file to a Whisper model running **on the same computer**. It implements the common `POST /v1/audio/transcriptions` request shape, so clients that can change their transcription API base URL can point to it. It does not capture audio from other apps or modify Wispr Flow.

## Real OpenAI Whisper calls

Open **http://127.0.0.1:8765/openai** for a plain working interface to OpenAI's `whisper-1` API. You can upload a file or record rolling six-second microphone clips. Each completed clip makes a separate real call to `https://api.openai.com/v1/audio/transcriptions`; it is **near-live chunked transcription**, not a persistent Realtime WebSocket. Text and errors come from the actual API response. The server keeps the key out of browser code.

After you add your own key, put `OPENAI_API_KEY=...` in this project's `.env.local` (which is gitignored) or set it in the server environment. The server reads `.env.local` on each request, so you can click **Check key again** without restarting. Do not paste your key into the browser page or commit it. The `/openai/status` endpoint reports only whether a key is configured, never its value. If no key is present, the cloud controls stay disabled and the local lab still works. OpenAI requests require internet and incur API charges; the local lab does neither.

The cloud endpoint is `POST /openai/transcribe` with multipart `file`, optional `language`, and optional `prompt`. It validates type and size, blocks cross-origin browser posts, and maps authentication, rate-limit, timeout, and provider errors to safe messages. It does not silently fall back to a demo transcript.

## Whisper Device Lab: same-model optimization POC

The browser lab at **http://127.0.0.1:8765/lab** can record from your microphone or accept a short audio file. It runs the **same selected Whisper checkpoint** as a full float32 baseline and an int8 variant, then applies a risk-directed precision controller. The controller scores int8 transcript spans using decoder log probability, repetitive output, and independent voice activity detection; it rechecks the highest-risk nonoverlapping spans in float32 within your chosen audio-duration budget. It shows the actual spans, reasons, before/after words, inference time, and peak process memory. If you paste a human-verified transcript, it also calculates word error rate for each run. The lab uses the device's microphone only after you click Record; the audio stays on this computer and the temporary upload is removed after comparison.

To launch it from this checkout:

```bash
source .venv/bin/activate
export LWB_MODEL_DIR="$PWD/models"
local-whisper-bridge serve
```

Then open the lab URL above. `models/tiny` is already prepared on this Mac. On another computer, first run `local-whisper-bridge prepare --model tiny` while online. Once weights are present, the comparison uses no cloud API. A direct API call is also available:

```bash
curl http://127.0.0.1:8765/lab/compare \
  -F file=@/absolute/path/to/audio.wav \
  -F reference='human verified words'
```

This is a **same-model adaptive-precision experiment**, not yet a deployable phone or Apple Watch app. The risk policy is implemented in `local_whisper_bridge/precision.py`; it is a heuristic proposed for this POC, not a proven novel research result or calibrated error detector. It can miss confidently wrong text; a float32 recheck can remain wrong or even be worse. CTranslate2 changes execution precision at load; the original checkpoint file is unchanged. Peak memory includes Python and runtime overhead and can be higher for int8 or adaptive runs. File-system cache and CPU temperature can affect sequential timings. A watch claim would require a native watchOS runtime and on-watch measurements of latency, memory, battery, and sustained recording.

The score is calculated from each real segment's decoder log probability, compression ratio, and no-speech probability, plus an independent voice-activity pass that finds speech with no transcript. Candidates are ranked by risk per second and rechecked within the selected target. A short recording can overrun the target by one indivisible risky utterance; the API reports that overrun. The application never inserts preset transcript text. The default target is 30%, and the UI slider lets you change it from 0% (no rechecks) to 100%.

The default model is multilingual `tiny` running on CPU with int8 quantization. The optional **coverage-aware adaptive repair** strategy makes a fast pass in 12-second windows. It compares the draft transcript against independent speech detection, spends its repair budget on speech that produced no text, then ranks uncertain spans by estimated benefit per second. Only those parts of the recording go through a second, larger model. This routing and budgeting logic is implemented here; CTranslate2's int8 inference itself comes from faster-whisper. There is no cloud fallback. Model weights are downloaded only by the explicit `prepare` command; after that, transcription can work with the network disconnected.

## Install and run (macOS, Linux, or Windows)

Use Python 3.10–3.13. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export LWB_MODEL_DIR="$PWD/models"
local-whisper-bridge prepare --model tiny
local-whisper-bridge serve
```

On Windows, use `py -3.12 -m venv .venv`, `.venv\Scripts\activate`, and `set LWB_MODEL_DIR=%CD%\models`. Model preparation needs internet once. The service listens only on `127.0.0.1:8765` and uses no external API key. Open `http://127.0.0.1:8765/docs` for interactive API documentation. The `models/tiny` and `models/base` weights have already been prepared on this computer, so you can skip `prepare` here.

## Call it from another app

```bash
curl http://127.0.0.1:8765/v1/audio/transcriptions \
  -F model=tiny \
  -F file=@/absolute/path/to/audio.mp3
```

The basic response is `{"text":"recognized words"}`. Add `-F response_format=verbose_json` for timestamped segments and detected language. Supported inputs: WAV, MP3, M4A, MP4, FLAC, OGG, and WebM, up to 25 MB. The upload is deleted from temporary storage after the request.

For adaptive repair, add `-F strategy=adaptive -F response_format=verbose_json`. The response includes `optimization.repair_seconds`, `optimization.repair_fraction`, `optimization.recovered_gaps`, and the source model for each segment. The default repair model is `base`; if its weights are absent, run `local-whisper-bridge prepare --model base` once. A larger repair model is possible with `LWB_REPAIR_MODEL=small` after preparing it.

The repair budget targets 25% of the file duration, with a 15-second minimum for short files and a 90-second cap; it never exceeds the file duration. The selection uses independent speech detection plus draft-model log probability, repetitive-text and speech-conflict signals. These are **heuristics**, not calibrated confidence or proof of correctness. A model can be confidently wrong; voice activity detection can miss speech or mistake noise for speech; slicing audio can lose context. Adaptive mode may be slower or less accurate than either model alone on some recordings. Benchmark it on representative, consented audio before using it in a product.

To compare warm-model latency and optional word error rate on your own recording:

```bash
python scripts/benchmark.py /absolute/path/to/audio.wav --reference 'human verified transcript'
```

On a 191-second synthetic recording on this Mac, adaptive mode ran the larger model on 11 seconds and took 7.5 seconds end-to-end, versus 12.3 seconds for the larger model on the full recording in one run. Its WER was 0.034 versus 0.000 for the full larger model; the fast model alone was faster and scored 0.006. On a 47-second mixed-quality test, adaptive mode reduced WER relative to the fast model but was slightly slower than the full larger model. These synthetic, single-run results show a latency/quality tradeoff, not a general performance guarantee.

Python clients using an OpenAI-style SDK can set `base_url="http://127.0.0.1:8765/v1"` and use any nonempty placeholder API key if their SDK requires one. This is a **compatible subset**, not a complete implementation of every OpenAI transcription parameter. An existing closed app must expose a configurable transcription endpoint or an integration API to use this bridge; it cannot be silently redirected.

## Change model or protect the endpoint

```bash
export LWB_MODEL=small
local-whisper-bridge prepare --model small
local-whisper-bridge serve
```

For a local bearer token, set `LWB_TOKEN` before starting the server and send `Authorization: Bearer <token>` in every transcription request. The health and model-list endpoints do not require the token. Never bind this development server to a public network interface.

`LWB_MODEL_DIR` changes where model weights live. `LWB_DEVICE=cpu` and `LWB_COMPUTE_TYPE=int8` are defaults; compatible accelerator configurations depend on the installed CTranslate2 build and hardware.

## Scope

This runs open Whisper-compatible CTranslate2 weights. It cannot run Wispr's Canto merely by selecting `model=canto`: that would require licensed model weights, an export to a supported inference format, a suitable runtime, and a first-party integration point. For products that already want this feature, see [Speaches](https://github.com/speaches-ai/speaches), [LocalAI](https://github.com/mudler/LocalAI), and [whisper.cpp](https://github.com/ggml-org/whisper.cpp). This project is a minimal integration prototype rather than a new inference engine.
