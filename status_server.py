"""LAN dashboard with live gameplay capture and training controls."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).parent.resolve()
DEFAULT_REPORT = ROOT / "data" / "training-progress.jsonl"
LIVE_PREVIEW_PATH = ROOT / "data" / "dashboard-live-screen.jpg"
LIVE_PREVIEW_TEMP = ROOT / "data" / "dashboard-live-screen.tmp.jpg"
PID_FILE = ROOT / "data" / "trainer.pid"
LOG_FILE = ROOT / "data" / "trainer-console.log"
LONGEST_RESET_FILE = ROOT / "data" / "longest-run-reset.json"
LONGEST_RESET_TEMP = ROOT / "data" / "longest-run-reset.tmp"
HTML = (ROOT / "status.html").read_bytes()
LOCK = threading.Lock()
TRAINER: subprocess.Popen | None = None
GAME_LAUNCH: subprocess.Popen | None = None
REPORT_CACHE: dict[str, object] = {
    "path": None, "offset": 0, "maximum": 0.0, "reset_at": None,
}
REPORT_LOCK = threading.Lock()


def trainer_pid() -> int | None:
    if TRAINER is not None and TRAINER.poll() is None:
        return TRAINER.pid
    try:
        pid = int(PID_FILE.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(bytes([0]), b" ").decode(errors="replace")
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except (OSError, ValueError):
        return None
    if "train_agent.py" in cmdline and cwd == ROOT:
        return pid
    return None


def game_running() -> bool:
    """Detect the Steam/Proton game process without running a shell command."""
    try:
        entries = Path("/proc").iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(bytes([0]), b" ").decode(errors="replace").lower()
            except OSError:
                continue
            if "vampiresurvivors.exe" in command or "vampire survivors.x86_64" in command:
                return True
    except OSError:
        pass
    return False

def launch_game() -> tuple[bool, str]:
    global GAME_LAUNCH
    with LOCK:
        if game_running():
            return False, "Vampire Survivors is already running."
        steam = shutil.which("steam") or next(
            (str(candidate) for candidate in (
                Path("/usr/games/steam"), Path("/usr/bin/steam"),
                Path("/usr/local/bin/steam"),
            ) if candidate.is_file() and os.access(candidate, os.X_OK)),
            None,
        )
        if not steam:
            return False, "Steam was not found on this VM."
        try:
            GAME_LAUNCH = subprocess.Popen(
                [steam, "steam://rungameid/1794680"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return False, f"Could not start Steam: {exc}"
    return True, "Steam was asked to launch Vampire Survivors. The game may take a moment to open."

def _recent_report_records(path: Path) -> list[dict]:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            stream.seek(max(0, end - 65536))
            lines = stream.read().splitlines()
    except OSError:
        return []
    records = []
    for line in reversed(lines):
        try:
            records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return records


def latest_report(path: Path) -> dict | None:
    for record in _recent_report_records(path):
        if record.get("event") in {"progress", "run_completed", "run_started", "session_stopped"}:
            return record
    return None


def latest_screen_status(path: Path) -> dict | None:
    for record in _recent_report_records(path):
        if record.get("event") == "screen_status":
            return record
    return None


def newest_preview_path(report_path: Path) -> Path | None:
    candidates = (LIVE_PREVIEW_PATH, report_path.with_name("current-screen.jpg"))
    available: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            available.append((candidate.stat().st_mtime_ns, candidate))
        except OSError:
            continue
    return max(available, default=(0, None), key=lambda item: item[0])[1]


def preview_timestamp(report_path: Path) -> str | None:
    path = newest_preview_path(report_path)
    if path is None:
        return None
    try:
        modified = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(modified, timezone.utc).isoformat()


def capture_live_screen() -> None:
    """Capture the visible game independently of the trainer's slower reports."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import cv2
        from linux_bridge import VampireSurvivorsBridge
    except Exception as exc:
        print(f"Live screen capture unavailable: {exc}", flush=True)
        return
    LIVE_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    bridge = None
    while True:
        try:
            if bridge is None:
                bridge = VampireSurvivorsBridge()
            frame = bridge.capture()
            height, width = frame.shape[:2]
            if width > 960:
                frame = cv2.resize(
                    frame, (960, round(height * 960 / width)),
                    interpolation=cv2.INTER_AREA,
                )
            if not cv2.imwrite(
                str(LIVE_PREVIEW_TEMP), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            ):
                raise OSError("Could not encode the live game frame")
            LIVE_PREVIEW_TEMP.replace(LIVE_PREVIEW_PATH)
        except Exception:
            if bridge is not None:
                try:
                    bridge.close()
                except Exception:
                    pass
            bridge = None
            time.sleep(.5)
            continue
        time.sleep(.25)

def quit_game() -> tuple[bool, str]:
    """Ask the visible game window to close cleanly; never kill its process."""
    if not game_running():
        return False, "Vampire Survivors is not running."
    xdotool = ROOT / ".local-deps" / "usr" / "bin" / "xdotool"
    if not xdotool.is_file():
        return False, "The bundled xdotool utility is missing; the game was left running."
    try:
        found = subprocess.run(
            [str(xdotool), "search", "--onlyvisible", "--name", "Vampire Survivors"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        window_ids = [line.strip() for line in found.stdout.splitlines() if line.strip().isdigit()]
        if not window_ids:
            return False, "Could not find the visible game window; the game was left running."
        closed = 0
        for window_id in window_ids:
            result = subprocess.run(
                [str(xdotool), "windowclose", window_id],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                closed += 1
        if closed:
            return True, "Close requested. Vampire Survivors should exit cleanly."
        return False, "The game window did not accept the close request; it was left running."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Could not request a clean close: {exc}"

def _complete_report_offset(path: Path) -> int:
    """Return a byte offset just after the last complete JSONL record."""
    try:
        size = path.stat().st_size
        start = max(0, size - 65536)
        with path.open("rb") as stream:
            stream.seek(start)
            tail = stream.read()
    except OSError:
        return 0
    newline = tail.rfind(b"\n")
    return start + newline + 1 if newline >= 0 else 0


def reset_longest_run(path: Path) -> tuple[bool, str]:
    """Reset the dashboard maximum without deleting training history."""
    with REPORT_LOCK:
        offset = _complete_report_offset(path)
        reset_at = datetime.now(timezone.utc).isoformat()
        marker = {
            "report_path": str(path),
            "offset": offset,
            "reset_at": reset_at,
        }
        try:
            LONGEST_RESET_FILE.parent.mkdir(parents=True, exist_ok=True)
            LONGEST_RESET_TEMP.write_text(
                json.dumps(marker, sort_keys=True), encoding="utf-8",
            )
            LONGEST_RESET_TEMP.replace(LONGEST_RESET_FILE)
        except OSError as exc:
            return False, f"Could not save longest-run reset: {exc}"
        REPORT_CACHE.update(
            path=str(path), offset=offset, maximum=0.0, reset_at=reset_at,
        )
    return True, "Longest run reset to 0:00. Completed run history was kept."


def maximum_completed_run(path: Path) -> float:
    """Incrementally track completed runs since the last dashboard reset."""
    try:
        size = path.stat().st_size
    except OSError:
        return float(REPORT_CACHE["maximum"])
    with REPORT_LOCK:
        reset_at = None
        reset_offset = 0
        try:
            marker = json.loads(LONGEST_RESET_FILE.read_text(encoding="utf-8"))
            if marker.get("report_path") == str(path):
                reset_at = marker.get("reset_at")
                reset_offset = max(0, int(marker.get("offset", 0)))
        except (OSError, ValueError, TypeError):
            pass
        if (
            REPORT_CACHE["path"] != str(path)
            or size < int(REPORT_CACHE["offset"])
            or REPORT_CACHE.get("reset_at") != reset_at
        ):
            REPORT_CACHE.update(
                path=str(path), offset=min(reset_offset, size),
                maximum=0.0, reset_at=reset_at,
            )
        offset = int(REPORT_CACHE["offset"])
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                data = stream.read()
        except OSError:
            return float(REPORT_CACHE["maximum"])
        complete_end = data.rfind(b"\n") + 1
        if complete_end:
            for line in data[:complete_end].splitlines():
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if record.get("event") == "run_completed":
                    try:
                        REPORT_CACHE["maximum"] = max(
                            float(REPORT_CACHE["maximum"]),
                            float(record.get("seconds", 0)),
                        )
                    except (TypeError, ValueError):
                        pass
            REPORT_CACHE["offset"] = offset + complete_end
        return float(REPORT_CACHE["maximum"])

def start_trainer(report: Path) -> tuple[bool, str]:
    global TRAINER
    with LOCK:
        if trainer_pid() is not None:
            return False, "Trainer is already running."
        local = ROOT / ".local-deps" / "usr"
        env = os.environ.copy()
        env["PATH"] = str(local / "bin") + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(local / "lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
        env["TESSDATA_PREFIX"] = str(local / "share/tesseract-ocr/5/tessdata")
        report.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("ab") as output:
            TRAINER = subprocess.Popen(
                [str(ROOT / ".venv/bin/python"), "-B", "-u", str(ROOT / "train_agent.py"),
                 "--full-run", "--deep-learning", "--report-path", str(report)],
                cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        PID_FILE.write_text(str(TRAINER.pid), encoding="utf-8")
        return True, "Training started."


def stop_trainer() -> tuple[bool, str]:
    pid = trainer_pid()
    if pid is None:
        return False, "Trainer is not running."
    try:
        os.killpg(pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        return False, "Could not signal the trainer."
    return True, "Stop requested. The trainer is saving its checkpoint."


class Handler(BaseHTTPRequestHandler):
    report_path = DEFAULT_REPORT
    def log_message(self, format: str, *args: object) -> None:
        print("[dashboard] " + (format % args), flush=True)
    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(HTML)
            return
        if route == "/api/status":
            diagnostic = latest_screen_status(self.report_path) or {}
            live_timestamp = preview_timestamp(self.report_path)
            if live_timestamp is not None:
                diagnostic["screen_preview_updated_at"] = live_timestamp
            payload = json.dumps({
                "trainer_running": trainer_pid() is not None,
                "game_running": game_running(),
                "latest": latest_report(self.report_path),
                "diagnostic": diagnostic,
                "maximum_run_seconds": maximum_completed_run(self.report_path),
                "report_path": str(self.report_path),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if route == "/api/screen-stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_path = None
            last_mtime = None
            while True:
                try:
                    preview = newest_preview_path(self.report_path)
                    if preview is None:
                        time.sleep(.2)
                        continue
                    mtime = preview.stat().st_mtime_ns
                    if preview != last_path or mtime != last_mtime:
                        payload = preview.read_bytes()
                        part = (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(payload)).encode("ascii")
                            + b"\r\n\r\n" + payload + b"\r\n"
                        )
                        self.wfile.write(part)
                        self.wfile.flush()
                        last_path = preview
                        last_mtime = mtime
                    time.sleep(.1)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
        if route == "/api/screen-preview":
            try:
                preview = newest_preview_path(self.report_path)
                if preview is None:
                    raise FileNotFoundError
                payload = preview.read_bytes()
            except OSError:
                self.send_error(404, "Screen preview is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)
    def do_POST(self) -> None:
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.send_error(403)
            return
        route = urlparse(self.path).path
        if route == "/api/launch":
            ok, message = launch_game()
        elif route == "/api/start":
            ok, message = start_trainer(self.report_path)
        elif route == "/api/stop":
            ok, message = stop_trainer()
        elif route == "/api/quit-game":
            ok, message = quit_game()
        elif route == "/api/reset-longest":
            ok, message = reset_longest_run(self.report_path)
        else:
            self.send_error(404)
            return
        payload = json.dumps({"ok": ok, "message": message}).encode()
        self.send_response(200 if ok else 409)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    Handler.report_path = args.report_path
    threading.Thread(target=capture_live_screen, name="dashboard-live-screen", daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard listening on port {args.port}; report: {args.report_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
