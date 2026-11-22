"""Read a Vampire Survivors window and expose opt-in WASD movement control.

This module does not start the game or send input on import. The caller must
explicitly call ``set_movement`` to hold/release keys. It uses only built-in
Windows APIs plus the NumPy/OpenCV packages already installed locally.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from collections.abc import Iterable
import time

import cv2
import numpy as np


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

_SRCCOPY = 0x00CC0020
_BI_RGB = 0
_DIB_RGB_COLORS = 0
_INPUT_KEYBOARD = 1
_INPUT_MOUSE = 0
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_SW_RESTORE = 9
# Vampire Survivors' keyboard movement is WASD on this user's setup.
_VK = {"up": 0x57, "left": 0x41, "down": 0x53, "right": 0x44}
_VK_ESCAPE = 0x1B


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]


_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = wintypes.HWND
_user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
_user32.GetClientRect.restype = wintypes.BOOL
_user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
_user32.ClientToScreen.restype = wintypes.BOOL
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.ShowWindow.restype = wintypes.BOOL
_user32.SetForegroundWindow.argtypes = [wintypes.HWND]
_user32.SetForegroundWindow.restype = wintypes.BOOL
_user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]
_user32.SendInput.restype = wintypes.UINT
_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = wintypes.BOOL
_user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
_user32.MapVirtualKeyW.restype = wintypes.UINT
_user32.GetDC.argtypes = [wintypes.HWND]
_user32.GetDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.ReleaseDC.restype = ctypes.c_int
_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.BitBlt.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD,
]
_gdi32.BitBlt.restype = wintypes.BOOL
_gdi32.GetDIBits.argtypes = [
    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
    ctypes.c_void_p, ctypes.POINTER(_BITMAPINFO), wintypes.UINT,
]
_gdi32.GetDIBits.restype = ctypes.c_int
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteObject.restype = wintypes.BOOL
_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_gdi32.DeleteDC.restype = wintypes.BOOL


def _last_error(message: str) -> OSError:
    return ctypes.WinError(ctypes.get_last_error(), message)


class VampireSurvivorsBridge:
    """Capture the visible game client and hold/release WASD keys."""

    def __init__(self, window_title: str = "Vampire Survivors", *, use_scan_codes: bool = True) -> None:
        self.window_title = window_title
        self.use_scan_codes = use_scan_codes
        self.hwnd = _user32.FindWindowW(None, window_title)
        if not self.hwnd:
            raise RuntimeError(f"Window not found: {window_title!r}")
        self._held: set[str] = set()

    def capture(self) -> np.ndarray:
        """Return the current client-area image as a BGR NumPy array.

        This captures from the desktop, so another window covering the game
        can obscure the pixels. The game must remain visible during capture.
        """
        client = wintypes.RECT()
        if not _user32.GetClientRect(self.hwnd, ctypes.byref(client)):
            raise _last_error("GetClientRect failed")
        width, height = client.right - client.left, client.bottom - client.top
        if width <= 0 or height <= 0:
            raise RuntimeError("The game window has an empty client area")

        origin = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(self.hwnd, ctypes.byref(origin)):
            raise _last_error("ClientToScreen failed")

        screen_dc = _user32.GetDC(None)
        if not screen_dc:
            raise _last_error("GetDC failed")
        memory_dc = None
        bitmap = None
        old_bitmap = None
        try:
            memory_dc = _gdi32.CreateCompatibleDC(screen_dc)
            bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
            if not memory_dc or not bitmap:
                raise _last_error("Could not allocate a capture bitmap")
            old_bitmap = _gdi32.SelectObject(memory_dc, bitmap)
            if not _gdi32.BitBlt(
                memory_dc, 0, 0, width, height,
                screen_dc, origin.x, origin.y, _SRCCOPY,
            ):
                raise _last_error("BitBlt failed while capturing the game")
            _gdi32.SelectObject(memory_dc, old_bitmap)

            info = _BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height  # top-down rows
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = _BI_RGB
            raw = (ctypes.c_ubyte * (width * height * 4))()
            rows = _gdi32.GetDIBits(
                screen_dc, bitmap, 0, height, ctypes.byref(raw),
                ctypes.byref(info), _DIB_RGB_COLORS,
            )
            if rows != height:
                raise _last_error("GetDIBits returned an incomplete frame")
            bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        finally:
            if old_bitmap and memory_dc:
                _gdi32.SelectObject(memory_dc, old_bitmap)
            if bitmap:
                _gdi32.DeleteObject(bitmap)
            if memory_dc:
                _gdi32.DeleteDC(memory_dc)
            _user32.ReleaseDC(None, screen_dc)

    def _send_key(self, vk: int, down: bool) -> None:
        scan_code = _user32.MapVirtualKeyW(vk, 0) if self.use_scan_codes else 0
        flags = 0 if down else _KEYEVENTF_KEYUP
        if self.use_scan_codes:
            flags |= _KEYEVENTF_SCANCODE
            vk = 0
        event = _INPUT(
            type=_INPUT_KEYBOARD,
            union=_INPUTUNION(
                ki=_KEYBDINPUT(
                    wVk=vk,
                    wScan=scan_code,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        if _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT)) != 1:
            raise _last_error(f"Could not send {'key down' if down else 'key up'}")

    def is_foreground(self) -> bool:
        return _user32.GetForegroundWindow() == self.hwnd

    def activate(self) -> bool:
        """Ask Windows to restore and foreground the game window."""
        _user32.ShowWindow(self.hwnd, _SW_RESTORE)
        _user32.SetForegroundWindow(self.hwnd)
        return self.is_foreground()

    def set_movement(self, directions: Iterable[str]) -> None:
        """Hold the requested directions; accepted names are up/left/down/right.

        Input is refused unless the Vampire Survivors window is foreground.
        Call ``release_all`` when stopping the controller.
        """
        requested = {direction.lower() for direction in directions}
        invalid = requested - _VK.keys()
        if invalid:
            raise ValueError(f"Unsupported directions: {sorted(invalid)}")
        if not self.is_foreground():
            raise RuntimeError("Vampire Survivors is not foreground; refusing input")

        verify_hold = bool(requested) and not self._held
        for direction in self._held - requested:
            self._send_key(_VK[direction], down=False)
            self._held.remove(direction)
        for direction in requested - self._held:
            self._send_key(_VK[direction], down=True)
            self._held.add(direction)
        if verify_hold:
            time.sleep(.04)
            if not all(self.windows_key_down(direction) for direction in requested):
                self.release_all()
                raise RuntimeError(
                    "Windows did not register the held WASD keys. "
                    "Run the controller as your normal desktop user, outside a low-integrity sandbox."
                )

    def pause_game(self) -> None:
        """Press Escape in the foreground game after verifying active play."""
        if not self.is_foreground():
            return
        self._send_key(_VK_ESCAPE, down=True)
        try:
            time.sleep(.065)
        finally:
            self._send_key(_VK_ESCAPE, down=False)

    def click_client(self, x: float, y: float) -> None:
        """Click a normalized point in the client after confirming focus."""
        if not self.is_foreground():
            raise RuntimeError("Vampire Survivors is not foreground; refusing mouse input")
        client = wintypes.RECT()
        if not _user32.GetClientRect(self.hwnd, ctypes.byref(client)):
            raise _last_error("GetClientRect failed before click")
        width, height = client.right - client.left, client.bottom - client.top
        point = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(self.hwnd, ctypes.byref(point)):
            raise _last_error("ClientToScreen failed before click")
        sx = point.x + round(min(max(float(x), 0.0), 1.0) * width)
        sy = point.y + round(min(max(float(y), 0.0), 1.0) * height)
        if not _user32.SetCursorPos(sx, sy):
            raise _last_error("Could not position the mouse for a game click")

        def mouse(flags: int) -> None:
            event = _INPUT(
                type=_INPUT_MOUSE,
                union=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flags, 0, 0)),
            )
            if _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT)) != 1:
                raise _last_error("Could not send mouse click")

        mouse(_MOUSEEVENTF_LEFTDOWN)
        try:
            time.sleep(.06)
        finally:
            mouse(_MOUSEEVENTF_LEFTUP)

    def windows_key_down(self, direction: str) -> bool:
        """Report Windows' current key state, independently of game movement."""
        return bool(_user32.GetAsyncKeyState(_VK[direction]) & 0x8000)

    def key_down(self, direction: str) -> bool:
        """Platform-neutral alias used by the movement probe."""
        return self.windows_key_down(direction)

    def release_all(self) -> None:
        """Release any movement keys this bridge is holding."""
        for direction in tuple(self._held):
            self._send_key(_VK[direction], down=False)
            self._held.remove(direction)
