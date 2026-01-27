"""Football-aware clip window planning.

Transforms raw audio peaks into preliminary clip segments using the broadcast
timing rules defined in Step 3. Scene alignment and merging happen later in
the pipeline, so this module only concerns itself with picking reasonable
start/end timestamps around each peak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence

from .editor import ClipSegment

if TYPE_CHECKING:  # pragma: no cover
    from audio import PeakEvent
    from config import ClipWindowConfig


class ClipWindowPlanner:
    """Compute pre/post windows for each detected football moment."""

    _SUSTAINED_WINDOW_S = 15.0
    _BACK_TO_BACK_SLACK_S = 5.0
    _SET_PIECE_GAP_S = 20.0
    _SHOT_SCORE_THRESHOLD = 0.85
    _RESOLUTION_AFTER_SHOT_S = 9.0
    _RESOLUTION_AFTER_SET_PIECE_S = 7.0

    def __init__(self, config: "ClipWindowConfig") -> None:
        self.config = config

    def plan(
        self,
        peaks: Iterable["PeakEvent"],
        *,
        video_duration_s: float | None = None,
    ) -> List[ClipSegment]:
        """Return clip segments where each peak marks the end of the action."""

        ordered_peaks = sorted(peaks, key=lambda p: p.timestamp_s)
        segments: List[ClipSegment] = []

        for idx, peak in enumerate(ordered_peaks):
            prev_peak = ordered_peaks[idx - 1] if idx > 0 else None
            pre = self._interp(
                self.config.min_pre_peak_s,
                self.config.max_pre_peak_s,
                peak.score,
            )
            post = self._interp(
                self.config.min_post_peak_s,
                self.config.max_post_peak_s,
                peak.score,
            )

            start = max(0.0, peak.timestamp_s - pre)
            end = peak.timestamp_s + post
            end = self._extend_for_sustained_excitement(
                ordered_peaks, idx, end
            )
            end = self._apply_resolution_window(end, peak, prev_peak)

            if video_duration_s is not None:
                end = min(video_duration_s, end)

            if end <= start:
                continue

            self._merge_into_segments(
                segments,
                ClipSegment(start_s=start, end_s=end),
            )

        return segments

    # --- helpers ---------------------------------------------------------

    def _extend_for_sustained_excitement(
        self,
        peaks: Sequence["PeakEvent"],
        idx: int,
        current_end: float,
    ) -> float:
        """Extend post window if consecutive peaks keep crowd noise high."""

        anchor = peaks[idx]
        for nxt in peaks[idx + 1 :]:
            delta = nxt.timestamp_s - anchor.timestamp_s
            if delta > self._SUSTAINED_WINDOW_S:
                break

            if nxt.score + 1e-3 < anchor.score * 0.7:
                # Excitement clearly drops, stop extending.
                break

            # Treat chain reactions (blocked shots, rebounds) as one clip.
            extra_post = self._interp(
                self.config.min_post_peak_s,
                self.config.max_post_peak_s,
                max(anchor.score, nxt.score),
            )
            current_end = max(current_end, nxt.timestamp_s + extra_post)

        return current_end

    def _apply_resolution_window(
        self,
        end: float,
        peak: "PeakEvent",
        prev_peak: "PeakEvent" | None,
    ) -> float:
        """Guarantee enough footage to show the outcome of the play."""

        resolution = 0.0
        if peak.score >= self._SHOT_SCORE_THRESHOLD:
            resolution = max(resolution, self._RESOLUTION_AFTER_SHOT_S)

        if prev_peak is not None:
            gap = peak.timestamp_s - prev_peak.timestamp_s
            if gap <= self._SET_PIECE_GAP_S:
                resolution = max(resolution, self._RESOLUTION_AFTER_SET_PIECE_S)

        return max(end, peak.timestamp_s + resolution)

    def _merge_into_segments(
        self,
        segments: List[ClipSegment],
        new_segment: ClipSegment,
    ) -> None:
        if not segments:
            segments.append(new_segment)
            return

        last = segments[-1]
        if new_segment.start_s <= last.end_s + self._BACK_TO_BACK_SLACK_S:
            segments[-1] = ClipSegment(
                start_s=last.start_s,
                end_s=max(last.end_s, new_segment.end_s),
            )
            return

        segments.append(new_segment)

    @staticmethod
    def _interp(min_value: float, max_value: float, score: float) -> float:
        """Linearly interpolate between min/max using the peak score."""

        clamped = max(0.0, min(1.0, score))
        return min_value + (max_value - min_value) * clamped
