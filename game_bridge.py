"""Select the desktop control bridge for the current operating system."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    from windows_bridge import VampireSurvivorsBridge
elif sys.platform.startswith("linux"):
    from linux_bridge import VampireSurvivorsBridge
else:
    raise RuntimeError(f"Unsupported desktop platform: {sys.platform}")

__all__ = ["VampireSurvivorsBridge"]
