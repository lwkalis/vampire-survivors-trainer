"""Capture gameplay before, during, and after a sustained WASD input."""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from game_bridge import VampireSurvivorsBridge
from xp_reward import level_up_visible


def live_hud_visible(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    border = cv2.cvtColor(
        frame[:max(5, round(height * .006)), width // 4:3 * width // 4],
        cv2.COLOR_BGR2HSV,
    )
    gold = cv2.inRange(border, (6, 80, 130), (42, 255, 255))
    health = frame[round(.50 * height):round(.52 * height),
                   round(.48 * width):round(.52 * width)]
    red = cv2.inRange(health, (0, 0, 170), (65, 65, 255))
    return bool((gold > 0).mean(axis=1).max() > .8 and (red > 0).sum(axis=1).max() >= width * .01)


def scenery_gray(frame: np.ndarray, left: float, right: float) -> np.ndarray:
    height, width = frame.shape[:2]
    crop = frame[round(.18 * height):round(.83 * height),
                 round(left * width):round(right * width)]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wait-for-play", action="store_true")
    parser.add_argument("--virtual-key", action="store_true")
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    args = parser.parse_args()
    bridge = VampireSurvivorsBridge(use_scan_codes=not args.virtual_key)
    before = bridge.capture()
    if args.wait_for_play:
        deadline = time.monotonic() + 50
        print("Waiting for the live HUD and player health bar", flush=True)
        while not live_hud_visible(before) or level_up_visible(before):
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the playfield")
            time.sleep(0.10)
            before = bridge.capture()
    if not live_hud_visible(before) or level_up_visible(before):
        raise RuntimeError("The live HUD or player health bar is missing")

    try:
        bridge.set_movement((args.direction,))
        time.sleep(0.60)
        during = bridge.capture()
        key_held = bridge.key_down(args.direction)
        time.sleep(0.60)
        after = bridge.capture()
    finally:
        bridge.release_all()

    print(f"Bridge key state during hold: {'down' if key_held else 'up'}")
    args.output.mkdir(parents=True, exist_ok=True)
    for label, frame in (("before", before), ("during", during), ("after", after)):
        if not cv2.imwrite(str(args.output / f"{label}.png"), frame):
            raise RuntimeError(f"Failed to save {label} frame")
    # Compare scenery away from the player and the HUD. The camera should
    # move scenery in the opposite direction to player movement.
    for side, x0, x1 in (("left", .10, .42), ("right", .58, .90)):
        for label, frame in (("during", during), ("after", after)):
            baseline = scenery_gray(before, x0, x1)
            sample = scenery_gray(frame, x0, x1)
            window = cv2.createHanningWindow(baseline.shape[::-1], cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(baseline, sample, window)
            print(f"{label} {side}: dx={dx:+.1f}px dy={dy:+.1f}px confidence={response:.2f}")


if __name__ == "__main__":
    main()
