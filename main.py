"""Entry point for the football highlights generator.

At this stage only the project structure is in place. Concrete logic for
audio analysis, scene alignment, and video editing will land in subsequent
steps.
"""

from __future__ import annotations

from pathlib import Path

from config import Config


def run_pipeline(config: Config) -> None:
    """Execute the end-to-end highlight generation workflow.

    The actual implementation will be filled in as each step of the plan is
    completed. For now we only log the expected stages so downstream modules
    can be wired incrementally without breaking the entry point.
    """

    print("[1/6] Preparing audio analysis components…")
    print("[2/6] Detecting candidate highlight moments…")
    print("[3/6] Building football-aware clip windows…")
    print("[4/6] Aligning to scene boundaries…")
    print("[5/6] Merging related attacking phases…")
    print("[6/6] Rendering consolidated highlight reel…")


if __name__ == "__main__":
    run_pipeline(Config(input_video=Path("input/full_match.mp4")))
