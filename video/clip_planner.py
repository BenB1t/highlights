"""Football-aware clip window planning with penalty handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Sequence

import numpy as np
from numpy.typing import NDArray

from .contraction import ClipContractor
from .editor import ClipSegment
from .penalty_detector import PenaltyShootoutDetector

if TYPE_CHECKING:  # pragma: no cover
    from audio import PeakEvent
    from config import ClipWindowConfig


class ClipWindowPlanner:
    """Compute clip windows while treating penalties as special cases."""

    _PHASE_MERGE_WINDOW_S = 12.0
    _SET_PIECE_PAIR_WINDOW_S = 20.0
    _SET_PIECE_MIN_POST_S = 14.0
    _SET_PIECE_MAX_POST_S = 18.0
    _SET_PIECE_OUTCOME_BUFFER_S = 5.0
    _BACK_TO_BACK_SLACK_S = 2.0
    _SHOT_SCORE_THRESHOLD = 0.85
    _RESOLUTION_AFTER_SHOT_S = 6.0
    _EXCITEMENT_DROP_PADDING_S = 3.0
    _UNSURE_EXTENSION_S = 4.0
    _PENALTY_PRE_S = 8.0
    _PENALTY_POST_S = 6.0

    def __init__(
        self,
        config: "ClipWindowConfig",
        penalty_detector: PenaltyShootoutDetector | None = None,
    ) -> None:
        self.config = config
        self._penalty_detector = penalty_detector or PenaltyShootoutDetector()
        self._last_loudness: NDArray[np.float32] | None = None

    def plan(
        self,
        peaks: Iterable["PeakEvent"],
        *,
        video_duration_s: float | None = None,
        loudness_profile: NDArray[np.float32] | None = None,
    ) -> List[ClipSegment]:
        """Return ordered clip segments for open play and penalties."""

        ordered_peaks = sorted(peaks, key=lambda p: p.timestamp_s)
        if loudness_profile is not None:
            self._last_loudness = np.asarray(loudness_profile, dtype=np.float32)
        if not ordered_peaks:
            return []

        duration_hint = (
            video_duration_s
            if video_duration_s is not None
            else ordered_peaks[-1].timestamp_s
            + self.config.max_post_peak_s
            + self._EXCITEMENT_DROP_PADDING_S
        )

        penalty_peaks, open_play_peaks = self._penalty_detector.filter_penalty_peaks(
            ordered_peaks, duration_hint
        )

        segments = self._plan_open_play_segments(open_play_peaks, duration_hint)
        penalty_segments = [
            self._create_penalty_segment(peak, duration_hint) for peak in penalty_peaks
        ]
        segments.extend(penalty_segments)
        segments = [seg for seg in segments if seg.end_s > seg.start_s]
        segments.sort(key=lambda seg: seg.start_s)

        if self._last_loudness is not None:
            segments = self._contract_segments(segments, ordered_peaks)

        return segments

    # --- open play planning ----------------------------------------------

    def _plan_open_play_segments(
        self, peaks: Sequence["PeakEvent"], video_duration_s: float
    ) -> List[ClipSegment]:
        if not peaks:
            return []

        clusters = self._cluster_peaks(peaks)
        segments: List[ClipSegment] = []
        for cluster in clusters:
            start = self._compute_cluster_start(cluster)
            end = self._compute_cluster_end(cluster)
            end = max(end, cluster[-1].timestamp_s + self._EXCITEMENT_DROP_PADDING_S)
            end = min(video_duration_s, end)
            if end <= start:
                continue
            self._merge_into_segments(segments, ClipSegment(start_s=start, end_s=end))
        return segments

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
            max(
                0.0,
                peak.timestamp_s
                - self._interp(
                    self.config.min_pre_peak_s,
                    self.config.max_pre_peak_s,
                    peak.score,
                ),
            )
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

    # --- penalty helpers --------------------------------------------------

    def _create_penalty_segment(
        self, peak: "PeakEvent", video_duration_s: float
    ) -> ClipSegment:
        start = max(0.0, peak.timestamp_s - self._PENALTY_PRE_S)
        end = peak.timestamp_s + self._PENALTY_POST_S
        end = min(video_duration_s, end)
        return ClipSegment(start_s=start, end_s=end)

    # --- contraction ------------------------------------------------------

    def _contract_segments(
        self, segments: Sequence[ClipSegment], peaks: Sequence["PeakEvent"]
    ) -> List[ClipSegment]:
        if self._last_loudness is None:
            return list(segments)

        contractor = ClipContractor(self._last_loudness)
        contracted: List[ClipSegment] = []
        for segment in segments:
            anchors = [
                peak.timestamp_s
                for peak in peaks
                if segment.start_s <= peak.timestamp_s <= segment.end_s
            ]
            slices = contractor.contract(segment.start_s, segment.end_s, anchors)
            if not slices:
                contracted.append(segment)
                continue
            for start, end in slices:
                if end <= start:
                    continue
                contracted.append(ClipSegment(start_s=start, end_s=end))

        contracted.sort(key=lambda seg: seg.start_s)
        return contracted

    # --- misc -------------------------------------------------------------

    @staticmethod
    def _interp(min_value: float, max_value: float, score: float) -> float:
        """Linearly interpolate between min/max using the peak score."""

        clamped = max(0.0, min(1.0, score))
        return min_value + (max_value - min_value) * clamped
