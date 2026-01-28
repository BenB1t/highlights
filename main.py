"""Entry point for the football highlights generator.

At this stage only the project structure is in place. Concrete logic for
audio analysis, scene alignment, and video editing will land in subsequent
steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import av

from audio import AudioPeakDetector
from config import Config
from scenes import SceneAligner, SceneDetector
from video.clip_planner import ClipWindowPlanner
from video.editor import VideoEditor
from video.penalty_detector import PenaltyShootoutDetector

if TYPE_CHECKING:  # pragma: no cover
    from video.contraction import ClipContractor


def run_pipeline(config: Config) -> None:
    """Execute the end-to-end highlight generation workflow."""

    print("[0/6] Loading source video metadata…")
    video_duration = _probe_duration(config.input_video)

    print("[1/6] Detecting audio peaks with penalty pattern recognition…")
    audio_detector = AudioPeakDetector(config.audio)
    peaks = audio_detector.detect(config.input_video)
    loudness_profile = audio_detector.get_last_loudness_profile()
    if not peaks:
        raise RuntimeError("Audio analysis yielded no peaks; aborting highlights generation.")

    penalty_detector = PenaltyShootoutDetector()
    shootout_ranges = penalty_detector.detect_shootout_periods(
        peaks, video_duration
    )

    print("[2/6] Planning clips with dead-time contraction…")
    clip_planner = ClipWindowPlanner(
        config.clip_window, penalty_detector=penalty_detector
    )
    planned_segments = clip_planner.plan(
        peaks,
        video_duration_s=video_duration,
        loudness_profile=loudness_profile,
    )
    if not planned_segments:
        raise RuntimeError("Clip planner created zero segments; cannot proceed.")

    print("[3/6] Detecting broadcast scene boundaries…")
    scene_detector = SceneDetector(config.scenes)
    scene_boundaries = scene_detector.detect(config.input_video)

    print("[4/6] Snapping clips to scene cuts…")
    aligner = SceneAligner(scene_boundaries)
    aligned_segments = aligner.align(
        planned_segments,
        video_duration_s=video_duration,
        preserve_segments=shootout_ranges,
    )
    if not aligned_segments:
        raise RuntimeError("Scene alignment removed all segments; cannot proceed.")

    print("[5/6] Merging clips from the same attack…")
    editor = VideoEditor(config)
    merged_segments = editor.merge_segments(aligned_segments)
    if len(merged_segments) < 3:
        logging.warning("Only %d segments produced after merging; reel may feel sparse.", len(merged_segments))
    penalty_segments_count = sum(
        1
        for seg in merged_segments
        if any(seg.start_s <= end and seg.end_s >= start for start, end in shootout_ranges)
    )
    if shootout_ranges and penalty_segments_count < 5:
        logging.warning(
            "Penalty shootout detected (%d periods) but only %d merged segments cover them.",
            len(shootout_ranges),
            penalty_segments_count,
        )

    if not merged_segments:
        raise RuntimeError("No segments remain after merging; cannot export highlights.")

    print("[6/6] Rendering consolidated highlight reel…")
    output_path = editor.export(config.input_video, merged_segments)
    print(f"Highlights saved to {output_path}")


def _probe_duration(video_path: Path) -> float:
    """Return video duration in seconds using PyAV probing."""

    time_base = getattr(av, "time_base", 1e-6)
    if hasattr(time_base, "__float__"):
        time_base_value = float(time_base)
    else:
        time_base_value = 1e-6

    try:
        with av.open(str(video_path)) as container:
            if container.duration is not None:
                return float(container.duration * time_base_value)

            stream_durations = []
            for stream in container.streams:
                if stream.duration is None or stream.time_base is None:
                    continue
                stream_durations.append(float(stream.duration * stream.time_base))

            if stream_durations:
                return max(stream_durations)
    except av.AVError as exc:  # pragma: no cover - depends on media input
        raise RuntimeError(f"Unable to open video for probing: {video_path}") from exc

    return 0.0


if __name__ == "__main__":
    run_pipeline(Config(input_video=Path("input/full_match.mp4")))
