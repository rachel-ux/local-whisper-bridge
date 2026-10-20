"""Allocate a limited float32 budget to risky int8 transcript spans.

This is an experimental risk policy, not a calibrated confidence estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    risk: float
    reason: str
    segment_index: int | None


def rank_candidates(segments: list[dict], speech_spans: list[dict], duration: float) -> list[Candidate]:
    candidates: list[Candidate] = []
    for index, segment in enumerate(segments):
        start = max(0.0, float(segment["start"]) - 0.35)
        end = min(duration, float(segment["end"]) + 0.35)
        if end - start < 0.25:
            continue
        logprob = float(segment.get("avg_logprob", 0))
        compression = float(segment.get("compression_ratio", 1))
        silence = float(segment.get("no_speech_prob", 0))
        # Signals are continuous; stronger model uncertainty moves a span up.
        risk = max(0.0, -logprob - 0.25) + max(0.0, compression - 1.8) * 0.5 + silence * 0.4
        if risk > 0.15:
            signals = []
            if logprob < -0.4:
                signals.append("low decoder probability")
            if compression > 1.8:
                signals.append("repetitive output")
            if silence > 0.35:
                signals.append("speech/silence ambiguity")
            candidates.append(Candidate(start, end, round(risk, 3), ", ".join(signals) or "uncertain draft", index))

    # A second, independent audio test catches speech that produced no words.
    for speech in speech_spans:
        left, right = max(0.0, float(speech["start"])), min(duration, float(speech["end"]))
        if right - left < 0.35:
            continue
        overlap = sum(max(0.0, min(right, float(s["end"])) - max(left, float(s["start"]))) for s in segments)
        coverage = min(1.0, overlap / (right - left))
        if coverage < 0.4:
            candidates.append(Candidate(max(0, left - 0.2), min(duration, right + 0.2), round(1.2 - coverage, 3), "speech without transcript", None))
    return sorted(candidates, key=lambda item: (-item.risk / max(0.5, item.end - item.start), -item.risk))


def allocate(candidates: list[Candidate], budget_seconds: float) -> list[Candidate]:
    if budget_seconds <= 0:
        return []
    chosen: list[Candidate] = []
    remaining = budget_seconds
    for item in candidates:
        length = item.end - item.start
        if length > remaining or any(max(item.start, other.start) < min(item.end, other.end) for other in chosen):
            continue
        chosen.append(item)
        remaining -= length
    # The audio is indivisible at this stage. Permit one bounded-budget overrun
    # when otherwise no uncertain speech could ever be checked on a short clip.
    if not chosen and candidates:
        chosen.append(candidates[0])
    return sorted(chosen, key=lambda item: item.start)
