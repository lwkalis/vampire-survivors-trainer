"""Capture and control the visible Vampire Survivors window on Linux/X11."""

from __future__ import annotations

from collections.abc import Iterable
import os
import shutil
import subprocess
import time

import cv2
import numpy as np


_KEY = {"up": "w", "left": "a", "down": "s", "right": "d"}


class VampireSurvivorsBridge:
    """X11 window capture and held-key/click control through xdotool."""

    def __init__(self, window_title: str = "Vampire Survivors", *, use_scan_codes: bool = True) -> None:
        del use_scan_codes  # Kept for constructor compatibility with the Windows bridge.
        if os.environ.get("XDG_SESSION_TYPE", "x11").casefold() == "wayland":
            raise RuntimeError(
                "This bridge needs an X11 desktop session. Log into Linux Mint's "
                "Cinnamon (X11) session before running the learner."
            )
        missing = [name for name in ("wmctrl", "xdotool", "tesseract")
                   if shutil.which(name) is None]
        if missing:
            raise RuntimeError(
                "Missing desktop tools: " + ", ".join(missing) +
                ". Install them with: sudo apt install wmctrl xdotool tesseract-ocr"
            )
        try:
            import mss
        except ImportError as exc:
            raise ImportError("Linux screen capture requires mss: python -m pip install mss") from exc
        # Find a visible game window before opening an X11 capture connection.
        # If lookup fails, no display client is left behind by a partial init.
        self.window_title = window_title.casefold()
        self._held: set[str] = set()
        self.window_id = self._find_window()[0]
        self._mss = mss.mss()

    def close(self) -> None:
        """Release the X11 connection used by MSS."""
        capture = getattr(self, "_mss", None)
        if capture is not None:
            self._mss = None
            capture.close()

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        if check and result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Desktop command failed ({args[0]}): {detail}")
        return result

    def _find_window(self) -> tuple[str, dict[str, int]]:
        result = self._run("wmctrl", "-lG", check=False)
        matches: list[tuple[int, str, dict[str, int]]] = []
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=7)
            if len(parts) < 8 or self.window_title not in parts[7].casefold():
                continue
            try:
                x, y, width, height = map(int, parts[2:6])
            except ValueError:
                continue
            if width <= 0 or height <= 0:
                continue
            geometry = {"left": x, "top": y, "width": width, "height": height}
            matches.append((width * height, parts[0], geometry))
        if not matches:
            raise RuntimeError(
                f"Could not find a visible X11 window containing {self.window_title!r}. "
                "Start Vampire Survivors in the Linux VM first."
            )
        _, window_id, geometry = max(matches)
        return window_id, geometry

    def capture(self) -> np.ndarray:
        """Capture the game window as a BGR image."""
        self.window_id, geometry = self._find_window()
        bgra = np.asarray(self._mss.grab(geometry))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def is_foreground(self) -> bool:
        self.window_id, _ = self._find_window()
        result = self._run("xdotool", "getactivewindow", check=False)
        try:
            active = int(result.stdout.strip(), 0)
            target = int(self.window_id, 0)
        except ValueError:
            return False
        return active == target

    def activate(self) -> bool:
        self.window_id, _ = self._find_window()
        self._run("wmctrl", "-ia", self.window_id)
        self._run("xdotool", "windowactivate", "--sync", self.window_id, check=False)
        time.sleep(.15)
        return self.is_foreground()

    def set_movement(self, directions: Iterable[str]) -> None:
        requested = {direction.casefold() for direction in directions}
        invalid = requested - _KEY.keys()
        if invalid:
            raise ValueError(f"Unsupported directions: {sorted(invalid)}")
        if not self.is_foreground():
            raise RuntimeError("Vampire Survivors is not foreground; refusing input")
        released = self._held - requested
        pressed = requested - self._held
        if released or pressed:
            args = ["xdotool"]
            for direction in sorted(released):
                args.extend(("keyup", _KEY[direction]))
            for direction in sorted(pressed):
                args.extend(("keydown", _KEY[direction]))
            self._run(*args)
        self._held = requested

    def pause_game(self) -> None:
        if self.is_foreground():
            self._run("xdotool", "key", "Escape")

    def click_client(self, x: float, y: float) -> None:
        if not self.is_foreground():
            raise RuntimeError("Vampire Survivors is not foreground; refusing mouse input")
        _, geometry = self._find_window()
        px = round(min(max(float(x), 0.0), 1.0) * geometry["width"])
        py = round(min(max(float(y), 0.0), 1.0) * geometry["height"])
        # A repeated click may target the current cursor position.
        self._run("xdotool", "mousemove", "--window", self.window_id,
                  str(px), str(py), "sleep", "0.04",
                  "mousedown", "1", "sleep", "0.06", "mouseup", "1")

    def press_key(self, key: str) -> None:
        """Tap a supported menu key after confirming the game has focus."""
        key_name = {"enter": "Return", "space": "space"}.get(key.casefold())
        if key_name is None:
            raise ValueError(f"Unsupported menu key: {key}")
        if not self.is_foreground():
            raise RuntimeError("Vampire Survivors is not foreground; refusing key input")
        self._run("xdotool", "key", key_name)

    def key_down(self, direction: str) -> bool:
        """Return whether this bridge currently holds the requested movement key."""
        return direction.casefold() in self._held

    def windows_key_down(self, direction: str) -> bool:
        # Compatibility with the existing Windows-only movement probe.
        return self.key_down(direction)

    def release_all(self) -> None:
        if self._held:
            args = ["xdotool"]
            for direction in sorted(self._held):
                args.extend(("keyup", _KEY[direction]))
            self._run(*args, check=False)
            self._held.clear()
