"""LAN dashboard for live training progress with fixed start and stop controls."""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import learning_check

ROOT = Path(__file__).parent.resolve()
DEFAULT_REPORT = ROOT / "data" / "training-progress.jsonl"
PID_FILE = ROOT / "data" / "trainer.pid"
LOG_FILE = ROOT / "data" / "trainer-console.log"
HTML = (ROOT / "status.html").read_bytes()
LOCK = threading.Lock()
TRAINER: subprocess.Popen | None = None
GAME_LAUNCH: subprocess.Popen | None = None
REPORT_CACHE: dict[str, object] = {"file": None, "offset": 0, "maximum": 0.0}
REPORT_LOCK = threading.Lock()
PROGRESS_EVENTS = {"progress", "run_completed", "run_started", "session_stopped"}


def trainer_pid() -> int | None:
    if TRAINER is not None and TRAINER.poll() is None:
        return TRAINER.pid
    try:
        pid = int(PID_FILE.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(bytes([0]), b" ").decode(errors="replace")
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except (OSError, ValueError):
        return None
    if cwd == ROOT and any(script in cmdline for script in ("train_agent.py", "learning_check.py")):
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


def _report_record(line: bytes) -> dict | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def latest_reports(path: Path) -> tuple[dict | None, dict | None]:
    """Read the newest progress record and screen heartbeat in one pass."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            start = max(0, stream.tell() - 65536)
            stream.seek(max(0, start - 1))
            if start:
                stream.readline()
            lines = stream.read().split(b"\n")
    except OSError:
        return None, None
    latest = diagnostic = None
    # Skip the trailing segment, which may be a write still in progress.
    for line in reversed(lines[:-1]):
        record = _report_record(line)
        if record is None:
            continue
        event = record.get("event")
        if not isinstance(event, str):
            continue
        if latest is None and event in PROGRESS_EVENTS:
            latest = record
        elif diagnostic is None and event == "screen_status":
            diagnostic = record
        if latest is not None and diagnostic is not None:
            break
    return latest, diagnostic


def quit_game() -> tuple[bool, str]:
    """Ask the visible game window to close cleanly; never kill its process."""
    if not game_running():
        return False, "Vampire Survivors is not running."
    wmctrl = shutil.which("wmctrl")
    if not wmctrl:
        return False, "wmctrl was not found; the game was left running."
    try:
        found = subprocess.run(
            [wmctrl, "-l"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        window_ids = []
        for line in found.stdout.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) == 4 and fields[3].strip().casefold() == "vampire survivors":
                window_ids.append(fields[0])
        if not window_ids:
            return False, "Could not find the visible game window; the game was left running."
        for window_id in window_ids:
            result = subprocess.run(
                [wmctrl, "-ic", window_id],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                return True, "Close requested. Vampire Survivors should exit cleanly."
        return False, "The game window did not accept the close request; it was left running."
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Could not request a clean close: {exc}"


def maximum_completed_run(path: Path) -> float:
    """Incrementally track the longest recorded completed run in the JSONL log."""
    with REPORT_LOCK:
        try:
            with path.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (path, stat.st_dev, stat.st_ino)
                if REPORT_CACHE["file"] != identity or stat.st_size < int(REPORT_CACHE["offset"]):
                    REPORT_CACHE.update(file=identity, offset=0, maximum=0.0)
                offset = int(REPORT_CACHE["offset"])
                stream.seek(offset)
                maximum = float(REPORT_CACHE["maximum"])
                while line := stream.readline():
                    if not line.endswith(b"\n"):
                        break
                    offset += len(line)
                    if b"run_completed" not in line and b"\\u" not in line:
                        continue
                    record = _report_record(line)
                    if record is None or record.get("event") != "run_completed":
                        continue
                    try:
                        seconds = float(record.get("seconds", 0))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if math.isfinite(seconds):
                        maximum = max(maximum, seconds)
        except OSError:
            return 0.0
        REPORT_CACHE.update(offset=offset, maximum=maximum)
        return maximum


def start_trainer(report: Path, *, check: bool = False) -> tuple[bool, str]:
    global TRAINER
    with LOCK:
        if trainer_pid() is not None:
            return False, "Trainer is already running."
        if not os.access(ROOT / ".venv/bin/python", os.X_OK):
            return False, "Trainer environment is missing. Run bash setup_linux.sh first."
        if check and not learning_check.CHECKPOINT.is_file():
            return False, "The learning check needs a saved DQN checkpoint. Run training first."
        if check and not game_running():
            return False, "Launch Vampire Survivors before starting the learning check."
        try:
            report = report.absolute()
            if check:
                command = [str(ROOT / ".venv/bin/python"), "-B", "-u", str(ROOT / "learning_check.py")]
            else:
                report.parent.mkdir(parents=True, exist_ok=True)
                command = ["bash", str(ROOT / "run-local.sh"),
                           "--full-run", "--deep-learning", "--report-path", str(report)]
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("ab") as output:
                TRAINER = subprocess.Popen(
                    command,
                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as exc:
            return False, f"Could not start the trainer: {exc}"
        message = "Learning check started." if check else "Training started."
        try:
            PID_FILE.write_text(str(TRAINER.pid), encoding="utf-8")
        except OSError as exc:
            return True, f"{message} Could not save the trainer PID: {exc}"
        return True, message


def stop_trainer() -> tuple[bool, str]:
    pid = trainer_pid()
    if pid is None:
        return False, "Trainer is not running."
    try:
        os.killpg(pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError):
        return False, "Could not signal the trainer."
    return True, "Stop requested. The trainer is saving its checkpoint."


def active_check_report(check: dict | None, pid: int | None) -> Path | None:
    if check is None or check.get("status") != "running":
        return None
    if pid is None or check.get("pid") != pid:
        check["status"] = "stopped"
        return None
    return learning_check.phase_report(check)


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
            pid = trainer_pid()
            check = learning_check.latest_check()
            phase_report = active_check_report(check, pid)
            latest, diagnostic = latest_reports(phase_report or self.report_path)
            if phase_report is not None and latest is not None:
                check["completed_in_phase"] = latest.get("completed_runs", 0)
            payload = json.dumps({
                "trainer_running": pid is not None,
                "game_running": game_running(),
                "learning_check": check,
                "latest": latest,
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
        if route == "/api/screen-preview":
            try:
                report = active_check_report(learning_check.latest_check(), trainer_pid()) or self.report_path
                payload = report.with_name("current-screen.jpg").read_bytes()
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
        elif route == "/api/learning-check":
            ok, message = start_trainer(self.report_path, check=True)
        elif route == "/api/stop":
            ok, message = stop_trainer()
        elif route == "/api/quit-game":
            ok, message = quit_game()
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
    Handler.report_path = args.report_path.absolute()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard listening on port {args.port}; report: {args.report_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
