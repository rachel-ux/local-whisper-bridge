from __future__ import annotations

import argparse
import os
import sys

from .engine import SUPPORTED_MODELS, model_path, model_root, selected_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Local, loopback-only Whisper transcription bridge")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download model weights once for later offline use")
    prepare.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    serve = commands.add_parser("serve", help="Start the local API")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.command == "prepare":
        # Keep all model-download metadata beside the weights. This also works
        # when a client cannot write to the normal Hugging Face cache folder.
        os.environ.setdefault("HF_HOME", str(model_root() / ".huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from faster_whisper.utils import download_model

        name = args.model or selected_model()
        destination = model_path(name)
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} into {destination} ...")
        download_model(name, output_dir=str(destination))
        print(f"Ready: {destination}")
    else:
        if not 1 <= args.port <= 65535:
            parser.error("port must be between 1 and 65535")
        import uvicorn

        print("Listening on 127.0.0.1 only. Audio and inference stay on this computer.", file=sys.stderr)
        uvicorn.run("local_whisper_bridge.app:app", host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
