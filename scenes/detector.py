"""Scene detection built on top of PySceneDetect."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

if TYPE_CHECKING:  # pragma: no cover
    from config import SceneConfig


@dataclass(slots=True)
class SceneBoundary:
    """Captures a start/end pair returned by the scene detector."""

    start_s: float
    end_s: float


class SceneDetector:
    """Adapter around PySceneDetect for broadcast-friendly cuts."""

    def __init__(self, config: "SceneConfig") -> None:
        self.config = config

    def detect(self, video_path: Path) -> List[SceneBoundary]:
        """Return scene boundaries sorted by time."""

        detector = self._build_detector()

        with open_video(str(video_path)) as video:
            scene_manager = SceneManager()
            scene_manager.add_detector(detector)
            scene_manager.detect_scenes(video)

            scene_list = scene_manager.get_scene_list()
            if not scene_list:
                duration = getattr(video, "duration", None)
                if duration is None:
                    return []
                end_s = self._timecode_to_seconds(duration)
                return [SceneBoundary(start_s=0.0, end_s=end_s)]

        boundaries = [
            SceneBoundary(
                start_s=self._timecode_to_seconds(start),
                end_s=self._timecode_to_seconds(end),
            )
            for start, end in scene_list
        ]

        return sorted(boundaries, key=lambda b: b.start_s)

    # --- helpers ---------------------------------------------------------

    def _build_detector(self):
        detector_type = self.config.detector.lower()
        if detector_type == "content":
            return ContentDetector(threshold=self.config.threshold)
        raise ValueError(f"Unsupported scene detector '{self.config.detector}'")

    @staticmethod
    def _timecode_to_seconds(timecode) -> float:
        if hasattr(timecode, "get_seconds"):
            return float(timecode.get_seconds())
        return float(timecode)
