"""Budgeted second-pass planning; no inference code lives in this module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepairCandidate:
    index: int
    seconds: float
    risk: float
    reason: str


@dataclass(frozen=True)
class CoverageGap:
    start: float
    end: float
    voiced_seconds: float


def repair_budget(duration: float) -> float:
    return min(duration, 90.0, max(15.0, duration * 0.25))


def plan_coverage_gaps(
    segments: list[dict], speech_spans: list[dict], duration: float, budget_seconds: float
) -> list[CoverageGap]:
    """Find transcript holes that overlap independently detected speech."""
    if duration <= 0 or budget_seconds <= 0:
        return []
    covered = sorted((max(0.0, float(s["start"])), min(duration, float(s["end"]))) for s in segments)
    holes: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in covered:
        if start - cursor >= 1.0:
            holes.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= 1.0:
        holes.append((cursor, duration))

    possible: list[CoverageGap] = []
    for start, end in holes:
        overlaps = [
            (max(start, float(s["start"])), min(end, float(s["end"])))
            for s in speech_spans
            if min(end, float(s["end"])) > max(start, float(s["start"]))
        ]
        voiced = sum(right - left for left, right in overlaps)
        if voiced < 0.75:
            continue
        # For a long hole, center the allowed window around the first detected
        # voice instead of spending the whole budget on leading silence.
        window_start = start
        if end - start > budget_seconds:
            window_start = min(max(start, overlaps[0][0] - 1.0), end - budget_seconds)
        possible.append(CoverageGap(window_start, min(end, window_start + budget_seconds), voiced))

    possible.sort(key=lambda gap: gap.voiced_seconds, reverse=True)
    chosen: list[CoverageGap] = []
    remaining = budget_seconds
    for gap in possible:
        seconds = min(gap.end - gap.start, remaining)
        if seconds >= 1.0:
            chosen.append(CoverageGap(gap.start, gap.start + seconds, gap.voiced_seconds))
            remaining -= seconds
    return sorted(chosen, key=lambda gap: gap.start)


def plan_repairs(
    segments: list[dict],
    duration: float,
    *,
    logprob_threshold: float = -0.45,
    budget_fraction: float = 0.25,
    minimum_budget_seconds: float = 15.0,
    maximum_budget_seconds: float = 90.0,
    budget_seconds: float | None = None,
) -> list[RepairCandidate]:
    """Choose uncertain spans by expected benefit per second of expensive compute.

    Scores are heuristics, not calibrated error probabilities. A segment can be
    confidently wrong, so this does not replace evaluation against references.
    """
    if duration <= 0:
        return []
    budget = (
        min(duration, budget_seconds)
        if budget_seconds is not None
        else min(duration, maximum_budget_seconds, max(minimum_budget_seconds, duration * budget_fraction))
    )
    candidates: list[RepairCandidate] = []
    for index, segment in enumerate(segments):
        seconds = max(0.0, float(segment["end"]) - float(segment["start"]))
        if seconds < 0.5 or seconds > budget:
            continue
        reasons: list[str] = []
        logprob = segment.get("avg_logprob")
        risk = 0.0
        if logprob is not None and logprob < logprob_threshold:
            risk += logprob_threshold - float(logprob)
            reasons.append("low_logprob")
        compression = segment.get("compression_ratio")
        if compression is not None and compression > 2.7:
            risk += (float(compression) - 2.7) / 2
            reasons.append("repetitive_text")
        if segment.get("temperature", 0) and float(segment["temperature"]) >= 0.4:
            risk += 0.2
            reasons.append("decoding_fallback")
        if segment.get("text", "").strip() and float(segment.get("no_speech_prob", 0)) > 0.6:
            risk += 0.2
            reasons.append("speech_conflict")
        if reasons:
            candidates.append(RepairCandidate(index, seconds, risk, ",".join(reasons)))

    candidates.sort(key=lambda candidate: candidate.risk / candidate.seconds, reverse=True)
    chosen: list[RepairCandidate] = []
    used = 0.0
    for candidate in candidates:
        if used + candidate.seconds <= budget + 1e-6:
            chosen.append(candidate)
            used += candidate.seconds
    return sorted(chosen, key=lambda candidate: candidate.index)
