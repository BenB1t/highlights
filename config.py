"""Centralised configuration for highlight generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class AudioConfig:
    """Parameters that drive loudness profiling and peak detection."""

    sample_rate: int = 22050
    smoothing_window_s: float = 3.0
    peak_percentile_low: float = 0.90
    peak_percentile_high: float = 0.97
    min_peak_gap_s: float = 60.0


@dataclass(slots=True)
class ClipWindowConfig:
    """Controls how far before/after a peak each clip spans."""

    min_pre_peak_s: float = 12.0
    max_pre_peak_s: float = 20.0
    min_post_peak_s: float = 6.0
    max_post_peak_s: float = 10.0


@dataclass(slots=True)
class SceneConfig:
    """Parameters for PySceneDetect boundary detection."""

    detector: str = "content"
    threshold: float = 27.0


@dataclass(slots=True)
class MergeConfig:
    """Rules governing whether neighbouring clips belong to one attack."""

    max_gap_s: float = 25.0


@dataclass(slots=True)
class OutputConfig:
    """Rendering configuration for the final highlight reel."""

    format: str = "mp4"
    codec: str = "libx264"
    audio_codec: str = "aac"
    fps: int | None = None


@dataclass(slots=True)
class Config:
    """Aggregate application configuration."""

    input_video: Path
    output_video: Path = Path("output/highlights.mp4")
    temp_dir: Path = Path(".cache")
    audio: AudioConfig = field(default_factory=AudioConfig)
    clip_window: ClipWindowConfig = field(default_factory=ClipWindowConfig)
    scenes: SceneConfig = field(default_factory=SceneConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
