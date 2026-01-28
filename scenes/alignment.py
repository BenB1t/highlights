"""Scene-based clip alignment helpers."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable, List, Sequence, Tuple

from video.editor import ClipSegment

from .detector import SceneBoundary


class SceneAligner:
    """Snaps clip windows to the nearest PySceneDetect boundaries."""

    def __init__(self, scene_boundaries: Iterable[SceneBoundary]):
        self.boundaries = sorted(scene_boundaries, key=lambda b: b.start_s)
        self.cuts = self._build_cut_list(self.boundaries)

    _PRESERVE_TOLERANCE_S = 0.5

    def align(
        self,
        segments: Iterable[ClipSegment],
        *,
        video_duration_s: float | None = None,
        preserve_segments: Sequence[Tuple[float, float]] | None = None,
    ) -> List[ClipSegment]:
        """Snap each clip start/end to the previous/next scene cut.

        Penalty shootout ranges can be supplied via ``preserve_segments`` to
        keep their boundaries frame-accurate.
        """

        preserve_ranges = preserve_segments or []

        if not self.cuts:
            return [ClipSegment(start_s=s.start_s, end_s=s.end_s) for s in segments]

        aligned: List[ClipSegment] = []
        for seg in segments:
            if preserve_ranges and self._overlaps_preserve(
                seg.start_s, seg.end_s, preserve_ranges
            ):
                start, end = self._preserve_within_range(
                    seg, video_duration_s
                )
            else:
                start = self._snap_start(seg.start_s)
                snapped_end = self._snap_end(seg.end_s, video_duration_s)
                if snapped_end <= start:
                    end = min(seg.end_s, video_duration_s or seg.end_s)
                else:
                    end = snapped_end

            if video_duration_s is not None:
                end = min(end, video_duration_s)

            if end <= start:
                continue

            aligned.append(ClipSegment(start_s=start, end_s=end))

        return aligned

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _build_cut_list(boundaries: List[SceneBoundary]) -> List[float]:
        if not boundaries:
            return []

        cut_times = [b.start_s for b in boundaries]
        cut_times.append(boundaries[-1].end_s)

        # Remove near-duplicates (within tolerance) while preserving order.
        tolerance = 1e-3
        unique: List[float] = []
        for cut in cut_times:
            if unique and abs(cut - unique[-1]) < tolerance:
                continue
            unique.append(cut)
        return unique

    def _snap_start(self, timestamp: float) -> float:
        idx = bisect_right(self.cuts, timestamp) - 1
        if idx < 0:
            return max(0.0, self.cuts[0])
        return self.cuts[idx]

    def _snap_end(self, timestamp: float, video_duration_s: float | None) -> float:
        idx = bisect_left(self.cuts, timestamp)
        if idx < len(self.cuts):
            return self.cuts[idx]
        if video_duration_s is not None:
            return video_duration_s
        return self.cuts[-1]

    def _overlaps_preserve(
        self,
        seg_start: float,
        seg_end: float,
        preserve_segments: Sequence[Tuple[float, float]],
    ) -> bool:
        for start, end in preserve_segments:
            if seg_end >= start and seg_start <= end:
                return True
        return False

    def _preserve_within_range(
        self, segment: ClipSegment, video_duration_s: float | None
    ) -> Tuple[float, float]:
        start = max(0.0, segment.start_s)
        end = segment.end_s
        if video_duration_s is not None:
            end = min(end, video_duration_s)

        if not self.cuts:
            return start, end

        snapped_start = self._snap_start(segment.start_s)
        snapped_end = self._snap_end(segment.end_s, video_duration_s)

        if abs(snapped_start - segment.start_s) > self._PRESERVE_TOLERANCE_S:
            start = snapped_start
        if abs(snapped_end - segment.end_s) > self._PRESERVE_TOLERANCE_S:
            end = snapped_end

        return start, end
