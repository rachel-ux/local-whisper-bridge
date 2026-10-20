"""Run one Whisper configuration in an isolated process for honest memory metrics."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

from .engine import model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--compute-type", choices=("float32", "int8"), required=True)
    parser.add_argument("--language")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    start = time.perf_counter()
    model = WhisperModel(
        str(model_path(args.model)), device="cpu",
        compute_type=args.compute_type, local_files_only=True,
        cpu_threads=4, num_workers=1,
    )
    loaded = time.perf_counter()
    segments, info = model.transcribe(
        args.audio, language=args.language or None, beam_size=1,
        temperature=0.0, condition_on_previous_text=False,
    )
    parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    finished = time.perf_counter()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    print(json.dumps({
        "compute_type": args.compute_type,
        "model": args.model,
        "text": " ".join(parts),
        "language": info.language,
        "audio_seconds": round(info.duration, 3),
        "load_seconds": round(loaded - start, 3),
        "inference_seconds": round(finished - loaded, 3),
        "total_seconds": round(finished - start, 3),
        "peak_process_mb": round(peak_mb, 1),
    }))


if __name__ == "__main__":
    main()
