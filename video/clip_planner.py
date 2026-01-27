"""Football-aware clip window planning.

Transforms raw audio peaks into preliminary clip segments using the broadcast
timing rules defined in Step 3. Scene alignment and merging happen later in
the pipeline, so this module only concerns itself with picking reasonable
start/end timestamps around each peak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List

from .editor import ClipSegment

if TYPE_CHECKING:  # pragma: no cover
    from audio import PeakEvent
    from config import ClipWindowConfig


class ClipWindowPlanner:
    """Compute pre/post windows for each detected football moment."""

    def __init__(self, config: "ClipWindowConfig") -> None:
        self.config = config

    def plan(
        self,
        peaks: Iterable["PeakEvent"],
        *,
        video_duration_s: float | None = None,
    ) -> List[ClipSegment]:
        """Return clip segments where each peak marks the end of the action."""

        segments: List[ClipSegment] = []
        for peak in peaks:
            # Higher scoring peaks get longer build-up/reaction windows.
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
            if video_duration_s is not None:
                end = min(video_duration_s, end)

            if end <= start:
                continue

            segments.append(ClipSegment(start_s=start, end_s=end))

        return segments

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _interp(min_value: float, max_value: float, score: float) -> float:
        """Linearly interpolate between min/max using the peak score."""

        clamped = max(0.0, min(1.0, score))
        return min_value + (max_value - min_value) * clamped
