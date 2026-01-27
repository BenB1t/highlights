"""Video editing scaffolding for assembling highlight reels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

from moviepy.editor import VideoFileClip, concatenate_videoclips

from utils import ensure_dir


@dataclass(slots=True)
class ClipSegment:
    """Represents a fully resolved clip window ready for export."""

    start_s: float
    end_s: float


class VideoEditor:
    """Handles clip extraction, merging, and concatenation."""

    if TYPE_CHECKING:  # pragma: no cover
        from config import Config

    def __init__(self, config: "Config") -> None:
        self.config = config

    def merge_segments(self, segments: Iterable[ClipSegment]) -> List[ClipSegment]:
        """Merge clips belonging to the same attacking sequence."""

        ordered = sorted(segments, key=lambda s: s.start_s)
        merged: List[ClipSegment] = []
        for seg in ordered:
            if not merged:
                merged.append(seg)
                continue

            prev = merged[-1]
            gap = seg.start_s - prev.end_s
            # Keep one clip per attacking wave if the reset gap is short.
            if gap <= self.config.merge.max_gap_s:
                merged[-1] = ClipSegment(
                    start_s=prev.start_s,
                    end_s=max(prev.end_s, seg.end_s),
                )
            else:
                merged.append(seg)

        return merged

    def export(self, video_path: Path, segments: Iterable[ClipSegment]) -> Path:
        """Concatenate ordered clips and write the highlight reel."""

        ordered = sorted(segments, key=lambda s: s.start_s)
        if not ordered:
            raise ValueError("No clip segments provided for export")

        ensure_dir(self.config.output_video.parent)

        output_path = self.config.output_video
        output_cfg = self.config.output

        with VideoFileClip(str(video_path)) as base_clip:
            subclips = [base_clip.subclip(seg.start_s, seg.end_s) for seg in ordered]
            final = concatenate_videoclips(subclips, method="compose")
            try:
                fps = output_cfg.fps or base_clip.fps
                if fps is None:
                    raise ValueError("Unable to determine FPS for output video")
                write_kwargs = {
                    "codec": output_cfg.codec,
                    "audio_codec": output_cfg.audio_codec,
                    "fps": fps,
                }
                final.write_videofile(str(output_path), **write_kwargs)
            finally:
                final.close()
                for clip in subclips:
                    clip.close()

        return output_path
