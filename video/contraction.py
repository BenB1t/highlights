"""Clip contraction utilities based on per-second loudness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation


@dataclass(slots=True)
class ExcitementWindow:
    start_s: float
    end_s: float
    intensity: float


class ClipContractor:
    SILENCE_THRESHOLD = 0.15  # 15% of max loudness
    MIN_SEGMENT_S = 2.0
    BRIDGE_GAP_S = 1.5

    def __init__(self, loudness_profile: NDArray[np.float32], sample_rate: int = 1):
        """loudness_profile is per-second array from AudioPeakDetector."""

        self.loudness = np.asarray(loudness_profile, dtype=np.float32)
        if self.loudness.ndim != 1:
            raise ValueError("loudness_profile must be 1-D")
        if int(sample_rate) != 1:
            raise NotImplementedError(
                "ClipContractor currently expects 1 Hz loudness profiles"
            )
        self.sample_rate = 1

    def contract(
        self, start_s: float, end_s: float, anchor_peaks: Sequence[float]
    ) -> List[Tuple[float, float]]:
        """Return contracted (start, end) tuples covering exciting portions."""

        start_idx = max(0, int(np.floor(start_s)))
        end_idx = min(len(self.loudness), int(np.ceil(end_s)))
        if start_idx >= end_idx:
            return []

        segment_loudness = self.loudness[start_idx:end_idx]
        if segment_loudness.size == 0:
            return []

        max_loudness = float(segment_loudness.max(initial=0.0))
        if max_loudness <= 0.0:
            return [(start_s, end_s)]

        threshold = max_loudness * self.SILENCE_THRESHOLD
        mask = segment_loudness >= threshold

        dilation_radius = int(round(self.MIN_SEGMENT_S))
        if dilation_radius > 0:
            mask = binary_dilation(mask, iterations=dilation_radius)

        segments = self._mask_to_segments(mask, start_idx)
        if not segments:
            segments = [(start_idx, end_idx)]

        segments = self._ensure_anchor_coverage(segments, anchor_peaks, start_idx, end_idx)
        segments = self._merge_close_segments(segments)

        return [(max(start_s, s), min(end_s, e)) for s, e in segments]

    # --- Internal helpers -------------------------------------------------

    def _mask_to_segments(
        self, mask: NDArray[np.bool_], offset: int
    ) -> List[Tuple[float, float]]:
        if mask.size == 0:
            return []

        transitions = np.diff(mask.astype(np.int8))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1

        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, mask.size)

        segments = []
        for s, e in zip(starts, ends):
            start_time = float(offset + s)
            end_time = float(offset + e)
            segments.append((start_time, end_time))
        return segments

    def _ensure_anchor_coverage(
        self,
        segments: List[Tuple[float, float]],
        anchor_peaks: Sequence[float],
        start_idx: int,
        end_idx: int,
    ) -> List[Tuple[float, float]]:
        if not anchor_peaks:
            return segments

        expanded = segments[:]
        for peak in anchor_peaks:
            if peak < start_idx or peak > end_idx:
                continue
            peak_time = float(peak)
            coverage = self.MIN_SEGMENT_S / 2.0
            has_cover = False
            for idx, (seg_start, seg_end) in enumerate(expanded):
                if seg_start <= peak_time <= seg_end:
                    has_cover = True
                    expand_start = min(seg_start, peak_time - coverage)
                    expand_end = max(seg_end, peak_time + coverage)
                    expanded[idx] = (expand_start, expand_end)
                    break
            if not has_cover:
                expanded.append((peak_time - coverage, peak_time + coverage))

        expanded.sort(key=lambda seg: seg[0])
        bounded = []
        for seg_start, seg_end in expanded:
            bounded.append((max(start_idx, seg_start), min(end_idx, seg_end)))
        return bounded

    def _merge_close_segments(
        self, segments: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        if not segments:
            return []

        segments.sort(key=lambda seg: seg[0])
        merged = [segments[0]]
        for seg_start, seg_end in segments[1:]:
            last_start, last_end = merged[-1]
            if seg_start - last_end <= self.BRIDGE_GAP_S:
                merged[-1] = (last_start, max(last_end, seg_end))
            else:
                merged.append((seg_start, seg_end))
        return merged
