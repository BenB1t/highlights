"""Audio peak detection based on football broadcast cues."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import av
from av.audio.resampler import AudioResampler
import numpy as np

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
        self._last_loudness: np.ndarray | None = None
        self._penalty_mode: bool = False
        self._logger = logging.getLogger(__name__)

    def detect(self, video_path: Path) -> List[PeakEvent]:
        """Extract audio, score per-second loudness, and drop close duplicates."""

        samples, sample_rate = self._extract_audio(video_path)
        if samples.size == 0:
            self._last_loudness = np.array([], dtype=np.float32)
            return []

        loudness = self._compute_loudness_per_second(samples, sample_rate)
        if loudness.size == 0:
            self._last_loudness = np.array([], dtype=np.float32)
            return []

        self._last_loudness = loudness.copy()

        low_thr, high_thr = self._compute_thresholds(loudness)
        candidates = self._find_local_maxima(loudness, low_thr, high_thr)
        self._penalty_mode = self._detect_penalty_pattern(candidates)

        gap = (
            self.config.penalty_gap_s
            if self._penalty_mode
            else getattr(self.config, "open_play_gap_s", self.config.min_peak_gap_s)
        )

        filtered = self._suppress_nearby_peaks(candidates, gap)
        return [PeakEvent(timestamp_s=c.timestamp_s, score=c.score) for c in filtered]

    def get_last_loudness_profile(self) -> np.ndarray:
        """Return the most recent per-second loudness profile."""

        if self._last_loudness is None:
            return np.array([], dtype=np.float32)
        return self._last_loudness.copy()

    # --- Private helpers -------------------------------------------------

    def _extract_audio(self, video_path: Path) -> tuple[np.ndarray, int]:
        """Stream audio samples via PyAV and resample to mono float32."""

        sample_rate = self.config.sample_rate
        try:
            container = av.open(str(video_path))
        except av.AVError as exc:  # pragma: no cover - depends on corrupt files
            self._logger.warning("Failed to open video %s: %s", video_path, exc)
            return np.array([], dtype=np.float32), sample_rate

        with container:
            audio_stream = next((s for s in container.streams.audio if s), None)
            if audio_stream is None:
                self._logger.info("No audio stream available in %s", video_path)
                return np.array([], dtype=np.float32), sample_rate

            resampler = AudioResampler(
                format="f32",
                layout="mono",
                rate=sample_rate,
            )

            chunks: list[np.ndarray] = []
            total_samples = 0

            try:
                for frame in container.decode(audio_stream):
                    self._append_resampled_frame(frame, resampler, chunks)
                self._flush_resampler(resampler, chunks)
            except av.AVError as exc:  # pragma: no cover - depends on corrupt files
                self._logger.warning(
                    "Error decoding audio stream for %s: %s", video_path, exc
                )
                return np.array([], dtype=np.float32), sample_rate

            if not chunks:
                return np.array([], dtype=np.float32), sample_rate

            total_samples = sum(chunk.size for chunk in chunks)
            samples = np.empty(total_samples, dtype=np.float32)
            offset = 0
            for chunk in chunks:
                end = offset + chunk.size
                samples[offset:end] = chunk
                offset = end

            return samples, sample_rate

    def _append_resampled_frame(
        self,
        frame: av.AudioFrame,
        resampler: AudioResampler,
        chunks: list[np.ndarray],
    ) -> None:
        resampled_frames = resampler.resample(frame) or []
        if not isinstance(resampled_frames, (list, tuple)):
            resampled_frames = [resampled_frames]

        for resampled in resampled_frames:
            chunk = resampled.to_ndarray()
            if chunk.ndim == 2:
                chunk = chunk[0]
            chunks.append(np.asarray(chunk, dtype=np.float32).copy())

    def _flush_resampler(
        self,
        resampler: AudioResampler,
        chunks: list[np.ndarray],
    ) -> None:
        flushed_frames = resampler.flush() or []
        if not isinstance(flushed_frames, (list, tuple)):
            flushed_frames = [flushed_frames]

        for frame in flushed_frames:
            if frame is None:
                continue
            chunk = frame.to_ndarray()
            if chunk.ndim == 2:
                chunk = chunk[0]
            chunks.append(np.asarray(chunk, dtype=np.float32).copy())

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

    def _suppress_nearby_peaks(
        self, candidates: Sequence[_PeakCandidate], gap_s: float | None = None
    ) -> List[_PeakCandidate]:
        """Keep only the loudest peak inside the supplied temporal gap."""

        if gap_s is None:
            gap = getattr(self.config, "open_play_gap_s", self.config.min_peak_gap_s)
        else:
            gap = gap_s

        ordered = sorted(candidates, key=lambda c: c.magnitude, reverse=True)
        selected: List[_PeakCandidate] = []
        for candidate in ordered:
            if all(abs(candidate.timestamp_s - s.timestamp_s) >= gap for s in selected):
                selected.append(candidate)

        selected.sort(key=lambda c: c.timestamp_s)
        return selected

    def _detect_penalty_pattern(
        self, candidates: Sequence[_PeakCandidate]
    ) -> bool:
        """Return True when peak spacing matches typical penalty cadence."""

        if len(candidates) < 4:
            return False

        timestamps = sorted(c.timestamp_s for c in candidates)
        intervals = np.diff(timestamps)
        if intervals.size < 3:
            return False

        lower_bound = 25.0
        upper_bound = 50.0
        tolerance = 6.0

        run_diffs: list[float] = []
        for interval in intervals:
            if lower_bound <= interval <= upper_bound:
                run_diffs.append(float(interval))
            else:
                if self._is_regular_penalty_run(run_diffs, tolerance):
                    return True
                run_diffs.clear()

        if self._is_regular_penalty_run(run_diffs, tolerance):
            return True
        return False

    def _is_regular_penalty_run(self, run: Sequence[float], tolerance: float) -> bool:
        if len(run) < 3:
            return False
        return max(run) - min(run) <= tolerance
