"""Compare warm-model latency and optional word error rate on one audio file."""

from __future__ import annotations

import argparse
import re
import time

from local_whisper_bridge.engine import LocalWhisperEngine, repair_model, selected_model


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    target, actual = words(reference), words(hypothesis)
    if not target:
        return 0.0 if not actual else 1.0
    row = list(range(len(actual) + 1))
    for i, word in enumerate(target, 1):
        next_row = [i]
        for j, candidate in enumerate(actual, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (word != candidate)))
        row = next_row
    return row[-1] / len(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--reference", help="Optional ground-truth transcript for WER")
    parser.add_argument("--reference-file", help="UTF-8 file containing the ground-truth transcript")
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()
    if args.reference and args.reference_file:
        parser.error("Use only one of --reference and --reference-file")
    reference = args.reference
    if args.reference_file:
        from pathlib import Path

        reference = Path(args.reference_file).read_text(encoding="utf-8")

    engine = LocalWhisperEngine()
    if not engine.ready() or not engine.repair_ready():
        parser.error("Prepare both the fast and repair models first")
    engine._get_model(selected_model())  # warm both models before timing
    engine._get_model(repair_model())
    for label, strategy in (("fast", "fast"), ("adaptive", "adaptive")):
        start = time.perf_counter()
        result = engine.transcribe(args.audio, None, None, strategy)
        elapsed = time.perf_counter() - start
        line = f"{label:8} {elapsed:.3f}s  repair_audio={result['optimization']['repair_seconds']:.1f}s"
        if reference:
            line += f"  WER={wer(reference, result['text']):.3f}"
        print(line)
        if args.show_text:
            print(f"  {result['text']}")

    bigger = engine._get_model(repair_model())
    start = time.perf_counter()
    segments, _ = bigger.transcribe(args.audio, beam_size=5, temperature=0.0)
    full_text = " ".join(segment.text.strip() for segment in segments).strip()
    elapsed = time.perf_counter() - start
    line = f"full {repair_model():<3} {elapsed:.3f}s  repair_audio=full file"
    if reference:
        line += f"  WER={wer(reference, full_text):.3f}"
    print(line)
    if args.show_text:
        print(f"  {full_text}")


if __name__ == "__main__":
    main()
