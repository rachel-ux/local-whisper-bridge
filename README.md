# Whisper Transcription Lab

One browser page with real OpenAI `whisper-1` calls and a separate, fully local Whisper inference path. OpenAI requests require internet and an API key. Offline requests use the downloaded open Whisper `tiny` checkpoint on this computer; audio is never sent to OpenAI in that mode.

The **custom quiet-gap rescue** algorithm is in `local_whisper_bridge/local_engine.py`. A first local pass uses CTranslate2 int8 CPU inference and voice activity filtering. The controller then measures short-window audio energy, finds low-volume regions with little or no transcript coverage, ranks them under a strict 20%-of-audio / 30-second retry budget, and reruns only those regions with less aggressive filtering and a wider beam. The returned response lists the retried time spans, any recovered segments, time spent, and real-time factor. This is a custom *inference policy*, not a new model or a proven accuracy improvement. Energy alone cannot reliably distinguish speech from noise, and a retry can still hallucinate or miss words.

## Run on a computer

```bash
cd /Users/rachelsequeira/Desktop/RandomCodingStuff/local-whisper-bridge
source .venv/bin/activate
local-whisper-bridge serve
```

Open http://127.0.0.1:8765/. Choose **Offline with quiet-gap rescue** to process without internet, **Offline baseline** to compare against the same model without retries, **OpenAI only** for cloud calls, or **Automatic** to use OpenAI first and fall back locally on a connection/timeout failure. The page has file upload and consecutive six-second microphone clips. If local inference takes longer than a clip, its transcript will lag behind recording.

On a new computer, install with `python -m pip install -e '.[test]'` and run `local-whisper-bridge prepare` once while online to download the model. This computer's `models/tiny` checkpoint is already prepared. Model preparation is an explicit command; offline transcription never downloads a model.

For OpenAI calls, provide `OPENAI_API_KEY` in the server environment or an ignored `.env.local` in this directory. The key stays on the server. OpenAI calls require internet and may incur charges. The project currently has no configured key, so live OpenAI calls have not been verified.

The server binds to `127.0.0.1`. `POST /offline/transcribe` accepts an audio file and an optional `rescue=false` query parameter for the baseline. `POST /openai/transcribe` forwards audio to OpenAI. Both reject files over 25 MB. Run tests with `python -m pytest -q`.

## Evidence and limits

On the OpenAI Whisper repository's 11-second JFK sample, both local modes returned the expected sentence; the rescue pass retried no spans. For a local 18-second test created by appending a 3-second, 3.5%-amplitude portion of that recording, the baseline omitted the appended quiet phrase. Quiet-gap rescue retried 13.05–15.25 seconds and recovered “And so, my fellow Americans.” Its end-to-end time on this Mac was about 1.5 seconds in one warmed run. This is one constructed example, not a quality or speed benchmark; test on consented whispered, noisy, multilingual, and overlapping-speaker recordings before making product claims.

This Python/FastAPI prototype runs on a computer. It does not run as an iPhone, Android, or Apple Watch app. Shipping the policy on those devices needs a native audio capture and local inference runtime, platform-specific packaging, and actual measurements of memory, battery, thermals, and accuracy. It also does not integrate with Wispr Flow or run Wispr's proprietary Canto model.
