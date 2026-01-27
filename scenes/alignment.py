"""Scene-based clip alignment helpers."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Iterable, List

from video.editor import ClipSegment

from .detector import SceneBoundary


class SceneAligner:
    """Snaps clip windows to the nearest PySceneDetect boundaries."""

    def __init__(self, scene_boundaries: Iterable[SceneBoundary]):
        self.boundaries = sorted(scene_boundaries, key=lambda b: b.start_s)
        self.cuts = self._build_cut_list(self.boundaries)

    def align(
        self,
        segments: Iterable[ClipSegment],
        *,
        video_duration_s: float | None = None,
    ) -> List[ClipSegment]:
        """Snap each clip start/end to the previous/next scene cut."""

        if not self.cuts:
            return [ClipSegment(start_s=s.start_s, end_s=s.end_s) for s in segments]

        aligned: List[ClipSegment] = []
        for seg in segments:
            snapped_start = self._snap_start(seg.start_s)
            snapped_end = self._snap_end(seg.end_s, video_duration_s)

            if video_duration_s is not None:
                snapped_end = min(snapped_end, video_duration_s)

            if snapped_end <= snapped_start:
                continue

            aligned.append(ClipSegment(start_s=snapped_start, end_s=snapped_end))

        return aligned

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _build_cut_list(boundaries: List[SceneBoundary]) -> List[float]:
        if not boundaries:
            return []

        cut_times = [b.start_s for b in boundaries]
        cut_times.append(boundaries[-1].end_s)

        # Remove duplicates while preserving chronological order.
        seen = set()
        unique: List[float] = []
        for cut in cut_times:
            if cut in seen:
                continue
            seen.add(cut)
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
