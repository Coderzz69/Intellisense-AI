#!/usr/bin/env python3
"""Generate a small synthetic intersection clip.

The repo ships no video, so this gives `python -m backend.server` something to
open out of the box. It exercises the whole chain (decode -> YOLO -> tracking ->
DQN); it is *not* a detection benchmark, since the detector is trained on real
traffic footage and will find little in flat synthetic shapes. Point
TRAFFIC_DEFAULT_VIDEO_PATH at a real clip to see meaningful counts.

    python scripts/make_sample_video.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "samples" / "synthetic_intersection.mp4"
ROAD = (90, 90, 90)
ASPHALT = 70


def render(width: int, height: int, fps: int, seconds: int, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {output}")

    mid_x, mid_y = width // 2, height // 2
    try:
        for index in range(fps * seconds):
            frame = np.full((height, width, 3), ASPHALT, np.uint8)
            cv2.rectangle(frame, (0, mid_y - 40), (width, mid_y + 40), ROAD, -1)
            cv2.rectangle(frame, (mid_x - 40, 0), (mid_x + 40, height), ROAD, -1)

            # A standing queue on the north approach.
            for slot in range(4):
                top = 20 + slot * 45
                cv2.rectangle(frame, (mid_x - 25, top), (mid_x + 25, top + 35), (40, 60, 200), -1)

            # One vehicle crossing east.
            x = (index * 6) % (width + 80) - 80
            cv2.rectangle(frame, (x, mid_y - 25), (x + 60, mid_y + 25), (200, 180, 40), -1)

            writer.write(frame)
    finally:
        writer.release()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()

    path = render(args.width, args.height, args.fps, args.seconds, args.output)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
