"""Entry point for the football highlights generator.

At this stage only the project structure is in place. Concrete logic for
audio analysis, scene alignment, and video editing will land in subsequent
steps.
"""

from __future__ import annotations

from pathlib import Path

from moviepy.editor import VideoFileClip

from audio import AudioPeakDetector
from config import Config
from scenes import SceneAligner, SceneDetector
from video import ClipWindowPlanner, VideoEditor


def run_pipeline(config: Config) -> None:
    """Execute the end-to-end highlight generation workflow."""

    print("[0/6] Loading source video metadata…")
    with VideoFileClip(str(config.input_video)) as clip:
        video_duration = float(clip.duration or 0.0)

    print("[1/6] Detecting audio-driven excitement peaks…")
    audio_detector = AudioPeakDetector(config.audio)
    peaks = audio_detector.detect(config.input_video)
    if not peaks:
        raise RuntimeError("Audio analysis yielded no peaks; aborting highlights generation.")

    print("[2/6] Building football-aware clip windows…")
    clip_planner = ClipWindowPlanner(config.clip_window)
    planned_segments = clip_planner.plan(peaks, video_duration_s=video_duration)
    if not planned_segments:
        raise RuntimeError("Clip planner created zero segments; cannot proceed.")

    print("[3/6] Detecting broadcast scene boundaries…")
    scene_detector = SceneDetector(config.scenes)
    scene_boundaries = scene_detector.detect(config.input_video)

    print("[4/6] Snapping clips to scene cuts…")
    aligner = SceneAligner(scene_boundaries)
    aligned_segments = aligner.align(planned_segments, video_duration_s=video_duration)
    if not aligned_segments:
        raise RuntimeError("Scene alignment removed all segments; cannot proceed.")

    print("[5/6] Merging clips from the same attack…")
    editor = VideoEditor(config)
    merged_segments = editor.merge_segments(aligned_segments)
    if not merged_segments:
        raise RuntimeError("No segments remain after merging; cannot export highlights.")

    print("[6/6] Rendering consolidated highlight reel…")
    output_path = editor.export(config.input_video, merged_segments)
    print(f"Highlights saved to {output_path}")


if __name__ == "__main__":
    run_pipeline(Config(input_video=Path("input/full_match.mp4")))
