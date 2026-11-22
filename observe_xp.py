"""Read one frame from the visible game window and report XP-bar fill."""

from __future__ import annotations

import argparse
import time

from game_bridge import VampireSurvivorsBridge
from xp_reward import XPRewardTracker, level_up_visible, xp_bar_fill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="sample continuously")
    parser.add_argument("--seconds", type=float, default=8.0, help="watch duration")
    parser.add_argument("--interval", type=float, default=0.2, help="sample period")
    args = parser.parse_args()

    bridge = VampireSurvivorsBridge()
    tracker = XPRewardTracker()
    start = time.monotonic()
    last = start
    deadline = start + (max(args.seconds, 0.1) if args.watch else 0.1)
    print("elapsed_s, xp_fill, level_up, reward")

    while True:
        frame = bridge.capture()
        now = time.monotonic()
        fill = xp_bar_fill(frame)
        level_up = level_up_visible(frame)
        reward = tracker.update(fill, now - last, level_up_seen=level_up)
        elapsed = now - start
        print(f"{elapsed:.2f}, {fill:.3f}, {level_up}, {reward.total:.4f}", flush=True)
        last = now
        if not args.watch or now >= deadline:
            break
        time.sleep(max(0.0, args.interval - (time.monotonic() - now)))

    height, width = frame.shape[:2]
    print(f"Captured client: {width}x{height}")


if __name__ == "__main__":
    main()
