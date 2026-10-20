from __future__ import annotations

import argparse
import os

from .local_engine import model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI Whisper call tester on this computer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Download the tiny Whisper model once for offline use")
    serve = commands.add_parser("serve", help="Start the OpenAI API proxy and browser page")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "prepare":
        os.environ.setdefault("HF_HOME", str(model_dir().parent / ".huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from faster_whisper.utils import download_model

        destination = model_dir()
        destination.mkdir(parents=True, exist_ok=True)
        download_model("tiny", output_dir=str(destination))
        print(f"Offline model ready: {destination}")
        return
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    import uvicorn

    uvicorn.run("local_whisper_bridge.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
