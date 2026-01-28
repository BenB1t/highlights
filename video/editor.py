"""Video editing scaffolding for assembling highlight reels."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Sequence

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
        self._logger = logging.getLogger(__name__)

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
        """Concatenate ordered clips via ffmpeg concat demuxer."""

        ordered = sorted(segments, key=lambda s: s.start_s)
        if not ordered:
            raise ValueError("No clip segments provided for export")

        ensure_dir(self.config.output_video.parent)

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            self._logger.warning(
                "ffmpeg binary not found; falling back to MoviePy exporter"
            )
            return self._export_with_moviepy(video_path, ordered)

        return self._export_with_ffmpeg(Path(ffmpeg_path), video_path, ordered)

    # --- FFmpeg path -----------------------------------------------------

    def _export_with_ffmpeg(
        self,
        ffmpeg_path: Path,
        video_path: Path,
        segments: Sequence[ClipSegment],
    ) -> Path:
        output_path = self.config.output_video
        output_cfg = self.config.output

        with tempfile.TemporaryDirectory(prefix="highlights_ffmpeg_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            concat_list_path = temp_dir_path / "segments.txt"
            segment_paths: List[Path] = []

            for idx, seg in enumerate(segments):
                duration = max(0.0, seg.end_s - seg.start_s)
                if duration <= 0:
                    continue
                seg_path = temp_dir_path / f"segment_{idx:04d}.mp4"
                self._logger.info(
                    "Encoding segment %d/%d (%.2f-%.2f s)",
                    idx + 1,
                    len(segments),
                    seg.start_s,
                    seg.end_s,
                )
                cmd = [
                    str(ffmpeg_path),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    f"{seg.start_s:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-c:v",
                    output_cfg.codec,
                    "-c:a",
                    output_cfg.audio_codec,
                ]
                if output_cfg.fps:
                    cmd += ["-r", str(output_cfg.fps)]
                cmd.append(str(seg_path))
                self._run_ffmpeg(cmd)
                segment_paths.append(seg_path)

            if not segment_paths:
                raise ValueError("All segments were empty; nothing to export")

            with concat_list_path.open("w", encoding="utf-8") as handle:
                for seg_path in segment_paths:
                    handle.write(f"file \"{seg_path.as_posix()}\"\n")

            concat_cmd = [
                str(ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",
            ]
            concat_cmd.append(str(output_path))
            self._run_ffmpeg(concat_cmd)

        return output_path

    def _run_ffmpeg(self, cmd: List[str]) -> None:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - external
            raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}") from exc

    # --- MoviePy fallback ------------------------------------------------

    def _export_with_moviepy(
        self, video_path: Path, segments: Sequence[ClipSegment]
    ) -> Path:
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "MoviePy fallback unavailable; please install moviepy or provide ffmpeg"
            ) from exc

        output_path = self.config.output_video
        output_cfg = self.config.output

        with VideoFileClip(str(video_path)) as base_clip:
            subclips = [base_clip.subclip(seg.start_s, seg.end_s) for seg in segments]
            final = concatenate_videoclips(subclips, method="compose")
            try:
                fps = output_cfg.fps or base_clip.fps
                if fps is None:
                    raise ValueError("Unable to determine FPS for output video")
                final.write_videofile(
                    str(output_path),
                    codec=output_cfg.codec,
                    audio_codec=output_cfg.audio_codec,
                    fps=fps,
                )
            finally:
                final.close()
                for clip in subclips:
                    clip.close()

        return output_path
