"""Audio peak detection based on football broadcast cues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import numpy as np
from moviepy.editor import VideoFileClip

if TYPE_CHECKING:  # pragma: no cover
    from config import AudioConfig


@dataclass(slots=True)
class PeakEvent:
    """Represents one audio-driven highlight candidate."""

    timestamp_s: float
    score: float


@dataclass(slots=True)
class _PeakCandidate:
    """Internal structure that tracks loudness for suppression logic."""

    timestamp_s: float
    score: float
    magnitude: float


class AudioPeakDetector:
    """Detect salient commentary/crowd peaks without GPU acceleration."""

    def __init__(self, config: "AudioConfig") -> None:
        self.config = config

    def detect(self, video_path: Path) -> List[PeakEvent]:
        """Extract audio, score per-second loudness, and drop close duplicates."""

        samples, sample_rate = self._extract_audio(video_path)
        if samples.size == 0:
            return []

        loudness = self._compute_loudness_per_second(samples, sample_rate)
        if loudness.size == 0:
            return []

        low_thr, high_thr = self._compute_thresholds(loudness)
        candidates = self._find_local_maxima(loudness, low_thr, high_thr)
        filtered = self._suppress_nearby_peaks(candidates)
        return [PeakEvent(timestamp_s=c.timestamp_s, score=c.score) for c in filtered]

    # --- Private helpers -------------------------------------------------

    def _extract_audio(self, video_path: Path) -> tuple[np.ndarray, int]:
        """Load mono audio samples using MoviePy."""

        with VideoFileClip(str(video_path)) as clip:
            audio = clip.audio
            if audio is None:
                return np.array([], dtype=np.float32), self.config.sample_rate

            # MoviePy handles resampling internally; we average channels to mono.
            sound_array = audio.to_soundarray(fps=self.config.sample_rate)

        if sound_array.ndim == 2:
            samples = sound_array.mean(axis=1)
        else:
            samples = sound_array.astype(np.float32)

        return np.asarray(samples, dtype=np.float32), self.config.sample_rate

    def _compute_loudness_per_second(
        self, samples: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Aggregate RMS loudness with optional smoothing."""

        seconds = int(np.ceil(len(samples) / sample_rate))
        if seconds == 0:
            return np.array([], dtype=np.float32)

        loudness = np.zeros(seconds, dtype=np.float32)
        for sec in range(seconds):
            start = sec * sample_rate
            end = min(start + sample_rate, len(samples))
            chunk = samples[start:end]
            if chunk.size == 0:
                continue
            # RMS captures perceived loudness better than raw amplitude.
            mono_chunk = chunk.astype(np.float32, copy=False)
            loudness[sec] = float(np.sqrt(np.mean(mono_chunk * mono_chunk)))

        window = max(1, int(round(self.config.smoothing_window_s)))
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            loudness = np.convolve(loudness, kernel, mode="same")

        return loudness

    def _compute_thresholds(self, loudness: np.ndarray) -> tuple[float, float]:
        """Return dynamic low/high percentile thresholds."""

        low_pct = np.clip(self.config.peak_percentile_low * 100.0, 0.0, 100.0)
        high_pct = np.clip(self.config.peak_percentile_high * 100.0, 0.0, 100.0)

        if high_pct < low_pct:
            low_pct, high_pct = high_pct, low_pct

        low_thr = float(np.percentile(loudness, low_pct))
        high_thr = float(np.percentile(loudness, high_pct))
        if np.isclose(high_thr, low_thr):
            high_thr = low_thr + 1e-6
        return low_thr, high_thr

    def _find_local_maxima(
        self, loudness: np.ndarray, low_thr: float, high_thr: float
    ) -> List[_PeakCandidate]:
        """Locate per-second peaks above percentile threshold."""

        candidates: List[_PeakCandidate] = []
        for idx in range(1, len(loudness) - 1):
            value = loudness[idx]
            if value < low_thr:
                continue
            if not (value >= loudness[idx - 1] and value > loudness[idx + 1]):
                continue
            score = min(1.0, (value - low_thr) / (high_thr - low_thr))
            candidates.append(
                _PeakCandidate(timestamp_s=float(idx), score=score, magnitude=value)
            )
        return candidates

    def _suppress_nearby_peaks(self, candidates: Sequence[_PeakCandidate]) -> List[_PeakCandidate]:
        """Keep the loudest peak within any 60 s window to avoid duplicates."""

        min_gap = self.config.min_peak_gap_s
        ordered = sorted(candidates, key=lambda c: c.magnitude, reverse=True)
        selected: List[_PeakCandidate] = []
        for candidate in ordered:
            if all(abs(candidate.timestamp_s - s.timestamp_s) >= min_gap for s in selected):
                selected.append(candidate)

        selected.sort(key=lambda c: c.timestamp_s)
        return selected
