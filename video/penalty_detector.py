"""Penalty shootout pattern detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from audio.peak_detection import PeakEvent


@dataclass(slots=True)
class PenaltyKick:
    """Lightweight structure to annotate detected penalties."""

    timestamp_s: float
    confidence: float
    is_goal: bool | None = None


class PenaltyShootoutDetector:
    """Detect penalty shootout spans based on peak cadence."""

    INTERVAL_S = 35.0
    INTERVAL_TOLERANCE = 10.0
    MIN_SEQUENCE_LENGTH = 4
    PRE_ROLL_S = 20.0
    POST_ROLL_S = 10.0

    def detect_shootout_periods(
        self, peaks: Sequence[PeakEvent], video_duration_s: float
    ) -> List[Tuple[float, float]]:
        """Find time ranges containing penalty shootouts using temporal pattern matching."""

        if not peaks:
            return []

        ordered = sorted(peaks, key=lambda p: p.timestamp_s)
        timestamps = np.array([p.timestamp_s for p in ordered], dtype=np.float32)

        if timestamps.size < self.MIN_SEQUENCE_LENGTH:
            return []

        intervals = np.diff(timestamps)
        lower = max(0.0, self.INTERVAL_S - self.INTERVAL_TOLERANCE)
        upper = self.INTERVAL_S + self.INTERVAL_TOLERANCE

        ranges: List[Tuple[float, float]] = []
        run_start_idx = 0
        run_length = 0

        for idx, gap in enumerate(intervals):
            if lower <= gap <= upper:
                if run_length == 0:
                    run_start_idx = idx
                run_length += 1
            else:
                if self._meets_sequence_requirement(run_length):
                    self._append_range(
                        timestamps,
                        run_start_idx,
                        run_start_idx + run_length,
                        video_duration_s,
                        ranges,
                    )
                run_length = 0

        if self._meets_sequence_requirement(run_length):
            self._append_range(
                timestamps,
                run_start_idx,
                run_start_idx + run_length,
                video_duration_s,
                ranges,
            )

        return self._merge_ranges(ranges)

    def is_penalty_period(
        self, timestamp_s: float, shootout_ranges: List[Tuple[float, float]]
    ) -> bool:
        """Check if timestamp falls within detected penalty period."""

        return any(start <= timestamp_s <= end for start, end in shootout_ranges)

    def filter_penalty_peaks(
        self, peaks: Sequence[PeakEvent], video_duration_s: float
    ) -> Tuple[List[PeakEvent], List[PeakEvent]]:
        """Return (penalty_peaks, open_play_peaks) separated."""

        shootout_ranges = self.detect_shootout_periods(peaks, video_duration_s)
        penalty_peaks: List[PeakEvent] = []
        open_play_peaks: List[PeakEvent] = []
        for peak in peaks:
            if self.is_penalty_period(peak.timestamp_s, shootout_ranges):
                penalty_peaks.append(peak)
            else:
                open_play_peaks.append(peak)

        return penalty_peaks, open_play_peaks

    # --- Internal helpers -------------------------------------------------

    def _append_range(
        self,
        timestamps: np.ndarray,
        start_idx: int,
        end_idx: int,
        video_duration_s: float,
        ranges: List[Tuple[float, float]],
    ) -> None:
        first_peak = float(timestamps[start_idx])
        last_idx = min(end_idx, len(timestamps) - 1)
        last_peak = float(timestamps[last_idx])
        start = max(0.0, first_peak - self.PRE_ROLL_S)
        end = min(video_duration_s, last_peak + self.POST_ROLL_S)
        ranges.append((start, end))

    def _meets_sequence_requirement(self, run_length: int) -> bool:
        # run_length counts intervals, so add one to compare against peak length
        return run_length + 1 >= self.MIN_SEQUENCE_LENGTH

    def _merge_ranges(
        self, ranges: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not ranges:
            return []

        ranges.sort(key=lambda r: r[0])
        merged: List[Tuple[float, float]] = [ranges[0]]
        for start, end in ranges[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged
