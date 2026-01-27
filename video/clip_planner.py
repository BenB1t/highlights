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

    _PHASE_MERGE_WINDOW_S = 25.0
    _SET_PIECE_PAIR_WINDOW_S = 20.0
    _SET_PIECE_MIN_POST_S = 14.0
    _SET_PIECE_MAX_POST_S = 18.0
    _SET_PIECE_OUTCOME_BUFFER_S = 5.0
    _BACK_TO_BACK_SLACK_S = 5.0
    _SHOT_SCORE_THRESHOLD = 0.85
    _RESOLUTION_AFTER_SHOT_S = 9.0
    _EXCITEMENT_DROP_PADDING_S = 6.0
    _UNSURE_EXTENSION_S = 4.0

    def __init__(self, config: "ClipWindowConfig") -> None:
        self.config = config

    def plan(
        self,
        peaks: Iterable["PeakEvent"],
        *,
        video_duration_s: float | None = None,
    ) -> List[ClipSegment]:
        """Return clip segments where each peak marks the trigger of an action."""

        ordered_peaks = sorted(peaks, key=lambda p: p.timestamp_s)
        if not ordered_peaks:
            return []

        clusters = self._cluster_peaks(ordered_peaks)
        segments: List[ClipSegment] = []

        for cluster in clusters:
            start = self._compute_cluster_start(cluster)
            end = self._compute_cluster_end(cluster)

            end = max(end, cluster[-1].timestamp_s + self._EXCITEMENT_DROP_PADDING_S)
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

    def _cluster_peaks(
        self, ordered_peaks: Sequence["PeakEvent"]
    ) -> List[List["PeakEvent"]]:
        clusters: List[List["PeakEvent"]] = []
        current: List["PeakEvent"] = []

        for peak in ordered_peaks:
            if not current:
                current.append(peak)
                continue

            gap = peak.timestamp_s - current[-1].timestamp_s
            if gap <= self._PHASE_MERGE_WINDOW_S:
                current.append(peak)
            else:
                clusters.append(current)
                current = [peak]

        if current:
            clusters.append(current)

        return clusters

    def _compute_cluster_start(self, cluster: Sequence["PeakEvent"]) -> float:
        start_candidates = [
            max(0.0,
                peak.timestamp_s
                - self._interp(
                    self.config.min_pre_peak_s,
                    self.config.max_pre_peak_s,
                    peak.score,
                ))
            for peak in cluster
        ]
        return min(start_candidates)

    def _compute_cluster_end(self, cluster: Sequence["PeakEvent"]) -> float:
        end_candidates = [
            peak.timestamp_s
            + self._interp(
                self.config.min_post_peak_s,
                self.config.max_post_peak_s,
                peak.score,
            )
            for peak in cluster
        ]

        end = max(end_candidates)

        if self._has_set_piece_signature(cluster):
            enforced_post = self._enforced_set_piece_post(cluster)
            end = max(end, cluster[-1].timestamp_s + enforced_post)
            end = max(end, cluster[-1].timestamp_s + self._SET_PIECE_OUTCOME_BUFFER_S)

        end = self._apply_resolution_window(end, cluster)
        end = self._prefer_longer_when_unsure(end, cluster)
        return end

    def _has_set_piece_signature(self, cluster: Sequence["PeakEvent"]) -> bool:
        for first, second in zip(cluster, cluster[1:]):
            if second.timestamp_s - first.timestamp_s <= self._SET_PIECE_PAIR_WINDOW_S:
                return True
        return False

    def _enforced_set_piece_post(self, cluster: Sequence["PeakEvent"]) -> float:
        top_score = max(peak.score for peak in cluster)
        return self._interp(
            self._SET_PIECE_MIN_POST_S,
            self._SET_PIECE_MAX_POST_S,
            top_score,
        )

    def _apply_resolution_window(
        self, end: float, cluster: Sequence["PeakEvent"]
    ) -> float:
        top_score = max(peak.score for peak in cluster)
        if top_score >= self._SHOT_SCORE_THRESHOLD:
            return max(end, cluster[-1].timestamp_s + self._RESOLUTION_AFTER_SHOT_S)
        return end

    def _prefer_longer_when_unsure(
        self, end: float, cluster: Sequence["PeakEvent"]
    ) -> float:
        top_score = max(peak.score for peak in cluster)
        if top_score < 0.5:
            return max(end, cluster[-1].timestamp_s + self._UNSURE_EXTENSION_S)
        return end

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
