"""Run the local screen-space dodging baseline against Vampire Survivors."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import difflib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid

import cv2

from scene_policy import MOVES, ScreenSpaceDodge, gameplay_hud_visible, pause_menu_visible
from game_bridge import VampireSurvivorsBridge
from xp_reward import (
    ChestPickupDetector, XPRewardTracker, health_bar_fill, level_up_visible,
    weapon_selection_grid_visible, six_choice_grid_visible,
    treasure_overlay_candidate, xp_bar_fill,
)
from upgrade_policy import (
    GoldCounter, KillCounter, UpgradeLearner, card_click_position, read_cards, read_game_over_heading,
    read_game_timer_seconds,
    read_screen_ocr, read_screen_text,
)


DATA_PATH = Path(__file__).parent / "data" / "vision_policy.json"
MENU_POLL_INTERVAL = .15
MENU_POST_CLICK_DELAY = .30
SCREEN_HEARTBEAT_INTERVAL = 2.0
SCREEN_PREVIEW_INTERVAL = 3.0
UNKNOWN_SCREEN_OCR_INTERVAL = 5.0


def write_progress_report(path: Path | None, event: str, **fields: object) -> None:
    if path is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def write_current_screen(path: Path | None, frame) -> str | None:
    """Atomically update the dashboard's local game-screen preview."""
    if path is None or frame is None:
        return None
    preview = path.with_name("current-screen.jpg")
    temporary = path.with_name("current-screen.tmp.jpg")
    try:
        if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 75]):
            return None
        temporary.replace(preview)
    except (OSError, cv2.error):
        return None
    return datetime.now(timezone.utc).isoformat()


class ScreenStatusReporter:
    """Publish a low-rate screen heartbeat and a throttled current-frame preview."""

    def __init__(self, report_path: Path | None) -> None:
        self.report_path = report_path
        self.phase: str | None = None
        self.phase_started = 0.0
        self.phase_since: str | None = None
        self.last_heartbeat = 0.0
        self.last_preview = 0.0
        self.preview_updated_at: str | None = None
        self.screen_text = ""

    def publish(self, frame, phase: str, *, screen_text: str | None = None) -> None:
        if self.report_path is None:
            return
        now = time.monotonic()
        changed = phase != self.phase
        if changed:
            self.phase = phase
            self.phase_started = now
            self.phase_since = datetime.now(timezone.utc).isoformat()
            self.screen_text = ""
        if screen_text is not None:
            cleaned = " ".join(screen_text.split())
            if cleaned:
                self.screen_text = cleaned[:800]
        if frame is not None and (changed or now - self.last_preview >= SCREEN_PREVIEW_INTERVAL):
            updated_at = write_current_screen(self.report_path, frame)
            if updated_at is not None:
                self.preview_updated_at = updated_at
            self.last_preview = now
        if changed or now - self.last_heartbeat >= SCREEN_HEARTBEAT_INTERVAL:
            write_progress_report(
                self.report_path, "screen_status", phase=phase,
                phase_since=self.phase_since,
                phase_duration_seconds=round(now - self.phase_started, 1),
                screen_text=self.screen_text,
                screen_preview_updated_at=self.preview_updated_at,
            )
            self.last_heartbeat = now


def load_stats(path: Path | None = None) -> tuple[list[int], list[float]]:
    path = DATA_PATH if path is None else path
    if not path.exists():
        return [0] * len(MOVES), [0.0] * len(MOVES)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            counts = saved.get("counts", [])
            means = saved.get("means", [])
            if len(counts) == len(MOVES) and len(means) == len(MOVES):
                counts = [int(value) for value in counts]
                means = [float(value) for value in means]
                if all(count >= 0 for count in counts) and all(map(math.isfinite, means)):
                    return counts, means
    except (OSError, ValueError, TypeError, OverflowError):
        pass
    return [0] * len(MOVES), [0.0] * len(MOVES)


def save_stats(counts: list[int], means: list[float], path: Path | None = None) -> None:
    path = DATA_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"counts": counts, "means": means, "actions": MOVES}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _label_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def startup_screen_phase(screen: str) -> str:
    if "interestedinthis" in screen:
        return "unlock-menu"
    if "photosensitivitywarning" in screen:
        return "photosensitivity-warning"
    if "gameover" in screen:
        return "game-over"
    if "characterselection" in screen:
        return "character-selection"
    if "stageselection" in screen:
        return "stage-selection"
    if "firstsurvivaton" in screen:
        return "title-splash"
    if "quickstart" in screen or "vampiresurvivors" in screen:
        return "title-menu"
    return "unknown-screen"


def click_ocr_label(
    bridge: VampireSurvivorsBridge, frame, label: str,
    fallback: tuple[float, float] | None = None,
    ocr: tuple[str, list[dict[str, object]]] | None = None,
) -> bool:
    """Click an OCR-matched menu label, using a known-layout fallback if needed."""
    text, lines = ocr if ocr is not None else read_screen_ocr(frame)
    wanted = _label_key(label)
    height, width = frame.shape[:2]
    for line in lines:
        if not isinstance(line, dict):
            continue
        actual = _label_key(str(line.get("text", "")))
        similar = difflib.SequenceMatcher(None, actual, wanted).ratio() >= .82
        prefix_match = wanted in {"quit", "options", "start"} and actual.startswith(wanted)
        if actual != wanted and not similar and not prefix_match:
            continue
        try:
            x = float(line["x"]) + float(line["width"]) / 2
            y = float(line["y"]) + float(line["height"]) / 2
        except (KeyError, TypeError, ValueError):
            x = y = 0.0
        if x <= 0 or y <= 0:
            if fallback is None:
                continue
            x, y = fallback[0] * width, fallback[1] * height
        bridge.click_client(x / width, y / height)
        return True
    if fallback is not None and wanted in _label_key(text):
        x, y = fallback
        bridge.click_client(x, y)
        return True
    return False


def stage_checkbox_checked(frame, stage: str) -> bool:
    """Read the known Character Selection stage checkboxes from their pixels."""
    centers = {"mad forest": (.753, .228), "inlaid library": (.753, .260)}
    center = centers.get(stage.casefold())
    if center is None:
        raise ValueError(f"Unknown stage checkbox: {stage}")
    height, width = frame.shape[:2]
    x, y = round(center[0] * width), round(center[1] * height)
    # Sample the inside of the checkbox, excluding the bright outline.
    radius = max(1, round(height * .006))
    crop = frame[max(0, y - radius):min(height, y + radius + 1),
                 max(0, x - radius):min(width, x + radius + 1), :3]
    return bool(crop.size and float((crop.max(axis=2) > 175).mean()) > .06)


def start_configured_run(
    bridge: VampireSurvivorsBridge, timeout: float, *, allow_active: bool = False,
    reporter: ScreenStatusReporter | None = None, require_fresh: bool = False,
) -> None:
    """Navigate title → Gennaro → Inlaid Library → gameplay, verifying each screen."""
    deadline = time.monotonic() + timeout
    title_start_clicked = False
    title_splash_clicked = False
    run_start_clicked = False
    character_clicked = False
    character_confirmed = False
    library_clicked = False
    mad_forest_clicked = False
    stage_start_clicked = False
    stage_start_last_attempt = 0.0
    unlock_back_last_attempt = 0.0
    last_message = None
    while time.monotonic() < deadline:
        if not bridge.is_foreground():
            message = "focus-required: click Vampire Survivors; waiting without input"
            if reporter is not None:
                reporter.publish(None, "focus-required")
        else:
            frame = bridge.capture()
            if ScreenSpaceDodge.game_over_visible(frame):
                screen = _label_key(read_screen_text(frame))
                if ("gameover" in screen
                        or ("over" in screen and "quit" in screen)):
                    print("startup: Game Over found; returning to title", flush=True)
                    if reporter is not None:
                        reporter.publish(frame, "game-over", screen_text=screen)
                    remaining = max(.1, deadline - time.monotonic())
                    return_to_title_after_death(bridge, remaining, reporter=reporter)
                    title_start_clicked = False
                    title_splash_clicked = False
                    run_start_clicked = False
                    character_clicked = False
                    character_confirmed = False
                    library_clicked = False
                    mad_forest_clicked = False
                    stage_start_clicked = False
                    last_message = None
                    continue
            if level_up_visible(frame):
                if require_fresh and not run_start_clicked:
                    raise RuntimeError("Evaluation needs a fresh run. Open the title or Game Over screen.")
                if reporter is not None:
                    reporter.publish(frame, "upgrade-menu")
                print("startup: open upgrade choice found; resuming the active run", flush=True)
                return
            if gameplay_hud_visible(frame):
                if reporter is not None:
                    reporter.publish(frame, "gameplay")
                if allow_active or run_start_clicked:
                    print("ready: configured Gennaro / Inlaid Library run is active", flush=True)
                    return
                raise RuntimeError(
                    "Gameplay is already active. Start the new run from the title menu "
                    "or use --attach to continue this run."
                )
            screen_text, screen_lines = read_screen_ocr(frame)
            screen = _label_key(screen_text)
            startup_labels = ("characterselection", "stageselection", "firstsurvivaton",
                              "vampiresurvivors", "quickstart", "gameover",
                              "photosensitivitywarning")
            if not any(label in screen for label in startup_labels):
                screen_text, screen_lines = read_screen_ocr(frame, white_text=True)
                screen = _label_key(screen_text)
            if reporter is not None:
                reporter.publish(frame, startup_screen_phase(screen), screen_text=screen_text)
            if "interestedinthis" in screen:
                # A run can be left on the collection's unlock/shop overlay.
                # It has no path into gameplay; return to the title menu first.
                now = time.monotonic()
                if now - unlock_back_last_attempt >= 1.5:
                    if not click_ocr_label(
                        bridge, frame, "Back", (.53, .84), (screen_text, screen_lines),
                    ):
                        bridge.click_client(.53, .84)
                    unlock_back_last_attempt = now
                    message = "startup: closing the unlock menu with Back"
                else:
                    message = "startup: waiting for the unlock menu to close"
                time.sleep(MENU_POST_CLICK_DELAY)
                continue
            if ("photosensitivity" in screen and "warning" in screen
                    and ("presstostart" in screen or "start" in screen)):
                # The first-launch disclaimer is a separate screen from the
                # ordinary title splash. Its button sits near the bottom
                # center of the game client (about 82% down the frame).
                bridge.click_client(.50, .82)
                message = "startup: acknowledged the photosensitivity warning"
                time.sleep(MENU_POST_CLICK_DELAY)
                continue
            if "gameover" in screen:
                print("startup: Game Over detected in OCR; returning to title", flush=True)
                remaining = max(.1, deadline - time.monotonic())
                return_to_title_after_death(bridge, remaining, reporter=reporter)
                title_start_clicked = False
                title_splash_clicked = False
                run_start_clicked = False
                character_clicked = False
                character_confirmed = False
                library_clicked = False
                mad_forest_clicked = False
                stage_start_clicked = False
                last_message = None
                continue
            if "characterselection" in screen:
                if not character_clicked:
                    if click_ocr_label(bridge, frame, "Gennaro", (.645, .28), (screen_text, screen_lines)):
                        character_clicked = True
                        message = "selection: chose Gennaro"
                    else:
                        message = "selection: waiting for the Gennaro card"
                elif not character_confirmed:
                    if not click_ocr_label(bridge, frame, "Confirm", (.824, .93), (screen_text, screen_lines)):
                        bridge.click_client(.824, .93)
                    character_confirmed = True
                    message = "selection: confirmed Gennaro"
                elif not library_clicked:
                    if not stage_checkbox_checked(frame, "Inlaid Library"):
                        bridge.click_client(.752, .261)
                        library_clicked = True
                        message = "selection: clicked Inlaid Library checkbox"
                    else:
                        library_clicked = True
                        message = "selection: Inlaid Library checked"
                elif (not mad_forest_clicked
                      and stage_checkbox_checked(frame, "Mad Forest")):
                    bridge.click_client(.752, .228)
                    mad_forest_clicked = True
                    message = "selection: unchecked Mad Forest"
                else:
                    start_button_clicked = click_ocr_label(
                        bridge, frame, "Start", (.824, .93), (screen_text, screen_lines),
                    )
                    if not start_button_clicked:
                        bridge.click_client(.824, .93)
                    run_start_clicked = True
                    message = "selection: starting Gennaro in Inlaid Library"
            elif "stageselection" in screen:
                if not library_clicked:
                    if click_ocr_label(bridge, frame, "Inlaid Library", (.365, .417), (screen_text, screen_lines)):
                        library_clicked = True
                        message = "selection: chose Inlaid Library"
                    else:
                        message = "selection: waiting for Inlaid Library"
                elif not stage_start_clicked:
                    # The Start label is sometimes omitted by Windows OCR on
                    # this busy stage screen; use the calibrated button point.
                    bridge.click_client(.824, .93)
                    stage_start_clicked = True
                    stage_start_last_attempt = time.monotonic()
                    run_start_clicked = True
                    message = "selection: starting Inlaid Library"
                elif time.monotonic() - stage_start_last_attempt >= 2.0:
                    # Retry if the menu did not accept the first click.
                    bridge.click_client(.824, .93)
                    stage_start_last_attempt = time.monotonic()
                    message = "selection: retrying Inlaid Library Start"
                else:
                    message = "selection: waiting for Inlaid Library to start"
            elif (not title_splash_clicked and "firstsurvivaton" in screen):
                bridge.click_client(.50, .65)
                title_splash_clicked = True
                message = "title-menu: advancing past the title splash"
            elif (not title_start_clicked
                  and ("quickstart" in screen or "vampiresurvivors" in screen)):
                # Tesseract often sees the title art but misses the large
                # central START button. The title-menu layout is stable, so
                # use its calibrated button center when OCR has no label box.
                if not click_ocr_label(
                    bridge, frame, "Start", (.50, .65), (screen_text, screen_lines),
                ):
                    bridge.click_client(.50, .65)
                title_start_clicked = True
                message = "title-menu: clicked Start"
            else:
                message = "menu-check: waiting for Start or the next selection screen"
        if message != last_message:
            print(message, flush=True)
            last_message = message
        time.sleep(MENU_POLL_INTERVAL)
    bridge.release_all()
    raise RuntimeError(
        f"The configured run did not reach gameplay within {timeout:g}s. "
        "Check that the game is focused and the expected selection screen is visible."
    )


def return_to_title_after_death(
    bridge: VampireSurvivorsBridge, timeout: float,
    *, reporter: ScreenStatusReporter | None = None,
) -> None:
    """Exit a dead run through Quit, using a revive only to reach run options."""
    deadline = time.monotonic() + timeout
    revive_clicked = False
    escape_tried = False
    paused_for_exit = False
    options_clicked = False
    quit_clicked = False
    splash_clicked = False
    last_message = None
    while time.monotonic() < deadline:
        if not bridge.is_foreground():
            message = "restart: focus the game; waiting without input"
            if reporter is not None:
                reporter.publish(None, "focus-required")
        else:
            frame = bridge.capture()
            text, lines = read_screen_ocr(frame)
            screen = _label_key(text)
            restart_labels = ("quickstart", "characterselection", "stageselection",
                              "survived", "firstsurvivaton", "options", "quit", "revive")
            if not any(label in screen for label in restart_labels):
                text, lines = read_screen_ocr(frame, white_text=True)
                screen = _label_key(text)
            ocr = (text, lines)
            if "characterselection" in screen or "stageselection" in screen:
                phase = "character-selection" if "characterselection" in screen else "stage-selection"
                if reporter is not None:
                    reporter.publish(frame, phase, screen_text=text)
                print("restart: reached selection for the next run", flush=True)
                return
            if ("quickstart" in screen
                    or ("vampiresurvivors" in screen and "start" in screen
                        and not ScreenSpaceDodge.game_over_visible(frame))):
                if reporter is not None:
                    reporter.publish(frame, "title-menu", screen_text=text)
                print("restart: returned to title menu", flush=True)
                return
            if "survived" in screen and ("done" in screen or "results" in screen):
                if reporter is not None:
                    reporter.publish(frame, "results-screen", screen_text=text)
                if click_ocr_label(bridge, frame, "Done", (.50, .935), ocr):
                    message = "restart: closed the Results screen"
                else:
                    bridge.click_client(.50, .935)
                    message = "restart: closing the Results screen"
                print(message, flush=True)
                time.sleep(MENU_POST_CLICK_DELAY)
                continue
            if "firstsurvivaton" in screen:
                if reporter is not None:
                    reporter.publish(frame, "title-splash", screen_text=text)
                height, width = frame.shape[:2]
                start_button = frame[round(.618 * height):round(.685 * height),
                                     round(.423 * width):round(.577 * width)]
                start_text, _ = read_screen_ocr(start_button, white_text=True, page_mode=7)
                if _label_key(start_text) == "start":
                    print("restart: returned to title menu", flush=True)
                    return
                if not splash_clicked:
                    bridge.click_client(.50, .65)
                    splash_clicked = True
                    print("restart: advancing past the title splash", flush=True)
                    time.sleep(MENU_POST_CLICK_DELAY)
                    continue
            pause_visible = pause_menu_visible(frame)
            over_heading = ""
            if ScreenSpaceDodge.game_over_visible(frame):
                over_heading = _label_key(read_game_over_heading(frame))
            if reporter is not None:
                if gameplay_hud_visible(frame):
                    phase = "gameplay"
                elif pause_visible:
                    phase = "pause-menu"
                elif ("gameover" in over_heading or "gameover" in screen
                      or ("over" in screen and "quit" in screen)):
                    phase = "game-over"
                else:
                    phase = "unknown-screen"
                reporter.publish(frame, phase, screen_text=text)
            if (("gameover" in over_heading or "gameover" in screen
                 or ("over" in screen and "quit" in screen))
                    and not (escape_tried and pause_visible)):
                if quit_clicked:
                    if (click_ocr_label(bridge, frame, "Yes", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Confirm", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Continue", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Quit", ocr=ocr)):
                        message = "restart: confirmed quitting the run"
                    else:
                        message = "restart: waiting for quit confirmation"
                elif click_ocr_label(bridge, frame, "Quit", ocr=ocr):
                    quit_clicked = True
                    message = "restart: clicked Quit on Game Over"
                elif "gameover" in over_heading:
                    height, width = frame.shape[:2]
                    quit_button = frame[round(.65 * height):round(.725 * height),
                                        round(.425 * width):round(.575 * width)]
                    quit_text, _ = read_screen_ocr(quit_button, white_text=True, page_mode=7)
                    if _label_key(quit_text) == "quit":
                        bridge.click_client(.50, .69)
                        quit_clicked = True
                        message = "restart: clicked Quit on Game Over"
                    elif not escape_tried:
                        bridge.pause_game()
                        escape_tried = True
                        message = "restart: checking whether Escape opens run options"
                    elif not revive_clicked and click_ocr_label(bridge, frame, "Revive", (.5, .866), ocr):
                        revive_clicked = True
                        message = "restart: using Revive to reach run options"
                    else:
                        message = "restart: waiting for Game Over controls"
                elif not escape_tried:
                    bridge.pause_game()
                    escape_tried = True
                    message = "restart: checking whether Escape opens run options"
                elif not revive_clicked and click_ocr_label(bridge, frame, "Revive", (.5, .866), ocr):
                    revive_clicked = True
                    message = "restart: using Revive to reach run options"
                else:
                    message = "restart: waiting for Game Over controls"
            elif gameplay_hud_visible(frame) and revive_clicked and not paused_for_exit:
                bridge.pause_game()
                paused_for_exit = True
                message = "restart: opened pause menu to quit the run"
            elif pause_visible:
                if not options_clicked and click_ocr_label(bridge, frame, "Options", ocr=ocr):
                    options_clicked = True
                    message = "restart: opened Options"
                elif options_clicked and not quit_clicked and click_ocr_label(bridge, frame, "Quit", ocr=ocr):
                    quit_clicked = True
                    message = "restart: clicked Quit in Options"
                elif quit_clicked:
                    if (click_ocr_label(bridge, frame, "Yes", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Confirm", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Continue", ocr=ocr)
                            or click_ocr_label(bridge, frame, "Quit", ocr=ocr)):
                        message = "restart: confirmed quitting the run"
                    else:
                        message = "restart: waiting for quit confirmation"
                else:
                    message = "restart: waiting for Options"
            elif quit_clicked or options_clicked:
                # The game can show a quit confirmation after the Options action.
                if (click_ocr_label(bridge, frame, "Yes", ocr=ocr)
                        or click_ocr_label(bridge, frame, "Confirm", ocr=ocr)
                        or click_ocr_label(bridge, frame, "Continue", ocr=ocr)):
                    message = "restart: confirmed quitting the run"
                elif not quit_clicked and click_ocr_label(bridge, frame, "Quit", ocr=ocr):
                    quit_clicked = True
                    message = "restart: clicked Quit in Options"
                elif click_ocr_label(bridge, frame, "Quit to Main Menu", ocr=ocr):
                    quit_clicked = True
                    message = "restart: clicked Quit to Main Menu"
                else:
                    message = "restart: waiting for Quit or confirmation"
            else:
                message = "restart: waiting for the post-death screen"
        if message != last_message:
            print(message, flush=True)
            last_message = message
        time.sleep(MENU_POLL_INTERVAL)
    bridge.release_all()
    raise RuntimeError(
        f"Could not return to the title menu within {timeout:g}s after Game Over."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", action="store_true", help="allow movement input")
    parser.add_argument("--full-run", action="store_true",
                        help="select Gennaro and Inlaid Library, learn, and restart after death")
    parser.add_argument("--attach", action="store_true",
                        help="with --full-run, accept an already active run as the first run")
    parser.add_argument("--deep-learning", action="store_true",
                        help="use a visual DQN movement policy (requires PyTorch)")
    parser.add_argument("--evaluate", action="store_true",
                        help="run a saved DQN with greedy movement and frozen learning")
    parser.add_argument("--checkpoint", type=Path,
                        default=DATA_PATH.parent / "visual_dqn_inlaid_library_8way.pt",
                        help="DQN checkpoint to load and save")
    parser.add_argument("--policy-dir", type=Path,
                        help="directory containing movement and upgrade policy data")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="stop after this many completed runs; 0 repeats until interrupted")
    parser.add_argument("--seconds", type=float, default=60.0, help="maximum live session")
    parser.add_argument("--action-seconds", type=float, default=0.24, help="time per move")
    parser.add_argument("--focus-timeout", type=float, default=120.0,
                        help="seconds to wait for game focus and active play")
    parser.add_argument("--evidence-interval", type=float, default=1.5,
                        help="seconds between gameplay evidence frames")
    parser.add_argument("--player-x", type=float, default=0.50, help="player anchor, normalized")
    parser.add_argument("--player-y", type=float, default=0.47, help="player anchor, normalized")
    parser.add_argument("--evidence-dir", type=Path, help="save periodic gameplay frames here")
    parser.add_argument("--report-path", type=Path,
                        default=Path(__file__).parent / "data" / "training-progress.jsonl",
                        help="append live JSONL training progress to this file")
    parser.add_argument("--pause-at-end", action="store_true", help="pause active play at session end")
    args = parser.parse_args()
    if args.evaluate:
        if args.attach:
            parser.error("--evaluate needs a fresh run and cannot use --attach")
        if not args.checkpoint.is_file():
            parser.error(f"evaluation checkpoint does not exist: {args.checkpoint}")
        args.full_run = args.deep_learning = True
    if args.full_run:
        args.control = True
        if args.evidence_dir is None:
            args.evidence_dir = Path(__file__).parent / "run-evidence" / uuid.uuid4().hex[:12]
        if args.evidence_interval == 1.5:
            args.evidence_interval = 6.0
    if args.attach and not args.full_run:
        parser.error("--attach requires --full-run")
    durations = (args.seconds, args.action_seconds, args.evidence_interval, args.focus_timeout)
    if any(not math.isfinite(value) or value <= 0 for value in durations) or args.max_runs < 0:
        parser.error("session, action, evidence, and focus durations must be positive; max-runs cannot be negative")
    if args.pause_at_end and not args.control:
        parser.error("--pause-at-end requires --control")
    learning_enabled = args.control and not args.evaluate
    movement_stats = args.policy_dir / DATA_PATH.name if args.policy_dir else DATA_PATH
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True, exist_ok=True)
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)

    screen_reporter = ScreenStatusReporter(args.report_path)
    screen_reporter.publish(None, "starting")
    bridge = VampireSurvivorsBridge()
    dqn = None
    dqn_state = None
    pending_dqn: tuple[object, int] | None = None
    if args.deep_learning:
        from dqn_policy import DQN_MOVE_INDICES, VisualDQN
        dqn = VisualDQN(args.checkpoint)
        epsilon = 0.0 if args.evaluate else dqn.epsilon
        print(f"deep-learning: visual DQN on {dqn.device}; epsilon={epsilon:.3f}; "
              f"steps={dqn.steps}; updates={dqn.updates}", flush=True)
    upgrades = UpgradeLearner(args.policy_dir)
    run_id = uuid.uuid4().hex[:12]
    completed_runs = 0
    write_progress_report(
        args.report_path, "session_started", run_id=run_id,
        deep_learning=args.deep_learning, evaluation=args.evaluate, max_runs=args.max_runs,
    )
    evidence_run_dir = (args.evidence_dir / run_id
                        if args.evidence_dir and args.full_run else args.evidence_dir)
    if evidence_run_dir:
        evidence_run_dir.mkdir(parents=True, exist_ok=True)
    run_picks: list[str] = []
    run_levels = 0
    run_chests = 0
    run_xp = 0.0
    counts, means = load_stats(movement_stats)
    policy = ScreenSpaceDodge(
        args.player_x, args.player_y,
        action_counts=counts, action_means=means,
    )
    tracker = XPRewardTracker()
    kill_counter = KillCounter()
    gold_counter = GoldCounter()
    chest_detector = ChestPickupDetector(player_x=args.player_x, player_y=args.player_y)
    treasure_prompt_seen = False
    treasure_click_at = float("-inf")
    treasure_text_checked_at = 0.0
    treasure_state_cache: str | None = None
    over_text_checked_at = 0.0
    over_state_cache = False

    def confirmed_level_up(frame) -> bool:
        # level_up_visible now matches the centered, machine-calibrated
        # "Level Up!" title itself. Windows OCR fragments that heading into
        # separate words on busy screens (e.g. "Level ... Up !"), so an OCR
        # text gate rejected genuine choices and let movement run underneath
        # the modal. Keep this visual confirmation as the authority.
        return level_up_visible(frame)

    def confirmed_game_over(frame) -> bool:
        nonlocal over_text_checked_at, over_state_cache
        if not ScreenSpaceDodge.game_over_visible(frame):
            over_text_checked_at = 0.0
            over_state_cache = False
            return False
        now = time.monotonic()
        if now - over_text_checked_at >= 1.5:
            heading = _label_key(read_game_over_heading(frame))
            over_state_cache = "gameover" in heading
            if not over_state_cache:
                text = _label_key(read_screen_text(frame))
                over_state_cache = ("gameover" in text
                                    or ("over" in text and "quit" in text))
            over_text_checked_at = now
        return over_state_cache

    def treasure_screen_state(frame) -> str | None:
        nonlocal treasure_text_checked_at, treasure_state_cache
        if not treasure_overlay_candidate(frame):
            treasure_state_cache = None
            treasure_text_checked_at = 0.0
            return None
        now = time.monotonic()
        if now - treasure_text_checked_at < 1.0:
            return treasure_state_cache
        treasure_text_checked_at = now
        text = read_screen_text(frame).lower()
        if "done" in text:
            treasure_state_cache = "done"
        elif "treasure found" in text and "open" in text:
            treasure_state_cache = "open"
        elif treasure_prompt_seen:
            treasure_state_cache = "animation"
        else:
            treasure_state_cache = None
        return treasure_state_cache

    def click_treasure_prompt(state: str) -> None:
        nonlocal treasure_click_at
        now = time.monotonic()
        if now - treasure_click_at < 1.0:
            return
        bridge.click_client(.50, .805)
        treasure_click_at = now
        print("treasure-found: opening the chest reward" if state == "open"
              else "treasure-reward: clicking Done", flush=True)

    def finish_session() -> None:
        write_progress_report(
            args.report_path, "session_stopped", completed_runs=completed_runs,
            dqn_steps=dqn.steps if dqn else None,
            dqn_updates=dqn.updates if dqn else None,
        )

    def save_dqn() -> None:
        dqn.save()
        print(f"deep-learning-saved: steps={dqn.steps}, updates={dqn.updates}, "
              f"checkpoint={dqn.checkpoint}", flush=True)

    def prepare_run() -> None:
        nonlocal run_chests, treasure_prompt_seen
        if not args.control:
            return
        bridge.activate()  # Best effort; foreground activation may need user help.
        if args.full_run:
            if not args.attach and bridge.is_foreground():
                initial_frame = bridge.capture()
                initial_text = _label_key(read_screen_text(initial_frame))
                if (confirmed_game_over(initial_frame)
                        or ("survived" in initial_text and ("done" in initial_text or "results" in initial_text))):
                    print("startup: post-run screen found; returning to title before a fresh run", flush=True)
                    return_to_title_after_death(
                        bridge, args.focus_timeout, reporter=screen_reporter,
                    )
            start_configured_run(
                bridge, args.focus_timeout, allow_active=args.attach,
                reporter=screen_reporter, require_fresh=args.evaluate,
            )
            return

        ready_deadline = time.monotonic() + args.focus_timeout
        waiting_message = None
        while True:
            if time.monotonic() >= ready_deadline:
                raise RuntimeError(
                    "Vampire Survivors did not reach foreground active play "
                    f"within {args.focus_timeout:g}s. Click the game window, "
                    "resume the run or choose its level-up card, then start again."
                )
            if not bridge.is_foreground():
                message = "focus-required: click the Vampire Survivors window; waiting without sending keys"
                screen_reporter.publish(None, "focus-required")
            else:
                ready_frame = bridge.capture()
                ready_level_up = confirmed_level_up(ready_frame)
                ready_treasure_state = (None if ready_level_up
                                        else treasure_screen_state(ready_frame))
                if ready_treasure_state is not None:
                    ready_phase = "treasure-reward"
                    if ready_treasure_state in {"open", "done"}:
                        if not treasure_prompt_seen:
                            run_chests += 1
                            treasure_prompt_seen = True
                        click_treasure_prompt(ready_treasure_state)
                    message = "treasure-reward: waiting for gameplay"
                elif confirmed_game_over(ready_frame):
                    raise RuntimeError(
                        "The visible run is at Game Over. Start or revive a run, "
                        "then launch the controller again."
                    )
                elif ready_level_up:
                    ready_phase = "upgrade-menu"
                    message = "level-up: choose a card; waiting without movement"
                elif pause_menu_visible(ready_frame):
                    ready_phase = "pause-menu"
                    message = "paused: resume the game; waiting without movement"
                elif gameplay_hud_visible(ready_frame):
                    treasure_prompt_seen = False
                    print("ready: game focused and active; controller starting", flush=True)
                    break
                else:
                    ready_phase = "unknown-screen"
                    message = "waiting: looking for active gameplay; no movement sent"
                screen_reporter.publish(ready_frame, ready_phase)
            if message != waiting_message:
                print(message, flush=True)
                waiting_message = message
            time.sleep(MENU_POLL_INTERVAL)

    with ExitStack() as cleanup:
        cleanup.callback(finish_session)
        if learning_enabled:
            cleanup.callback(save_stats, counts, means, movement_stats)
            if dqn is not None:
                cleanup.callback(save_dqn)
        if args.control:
            cleanup.callback(bridge.release_all)
        prepare_run()
        start = previous_time = time.monotonic()
        next_evidence = 0.0
        unknown_ocr_checked_at = 0.0
        unknown_ocr_text = ""
        unlock_back_last_attempt = 0.0
        action = None
        last_state = None
        focus_lost_since: float | None = None
        end_choice_deadline: float | None = None
        print("elapsed_s,xp_fill,health_fill,action,danger,kills,kill_reward,xp_reward,level_reward,chest_reward,lamp_reward,chicken_reward,damage_penalty,total,state", flush=True)
        while True:
            if args.control and not bridge.is_foreground():
                screen_reporter.publish(None, "focus-required")
                bridge.release_all()
                action = None
                if focus_lost_since is None:
                    dqn_state = pending_dqn = None
                    tracker.reset()
                    focus_lost_since = time.monotonic()
                    print("focus-lost: movement released; checking whether play needs pausing", flush=True)
                    lost_frame = bridge.capture()
                    lost_level_up = confirmed_level_up(lost_frame)
                    lost_over = not lost_level_up and confirmed_game_over(lost_frame)
                    lost_paused = (not (lost_level_up or lost_over)
                                   and pause_menu_visible(lost_frame))
                    lost_active = (not (lost_level_up or lost_over or lost_paused)
                                   and gameplay_hud_visible(lost_frame))
                    if lost_over:
                        print("game-over: run ended while focus was away", flush=True)
                        break
                    if lost_paused:
                        print("paused: leaving the game safely paused", flush=True)
                        break
                    if lost_active and bridge.activate():
                        bridge.pause_game()
                        print("focus-lost-paused: gameplay paused safely", flush=True)
                        break
                elif time.monotonic() - focus_lost_since >= 60:
                    raise RuntimeError("Vampire Survivors stayed out of focus for 60 seconds; movement stopped safely")
                time.sleep(.20)
                continue
            if focus_lost_since is not None:
                resumed_at = time.monotonic()
                start += resumed_at - focus_lost_since
                previous_time = resumed_at
                focus_lost_since = None
                last_state = None
            frame = bridge.capture()
            now = time.monotonic()
            elapsed = now - start
            level_up = confirmed_level_up(frame)
            treasure_state = (None if level_up else treasure_screen_state(frame))
            treasure = treasure_state is not None
            chest_prompt = treasure_state in {"open", "done"}
            chest_prompt_event = chest_prompt and not treasure_prompt_seen
            game_over = not (level_up or treasure) and confirmed_game_over(frame)
            paused = not (level_up or treasure or game_over) and pause_menu_visible(frame)
            active = not (level_up or treasure or game_over or paused) and gameplay_hud_visible(frame)
            if args.control and not active:
                bridge.release_all()
            if active or level_up or game_over:
                treasure_prompt_seen = False
                treasure_click_at = float("-inf")
            state = ("level-up" if level_up else "treasure" if treasure else
                     "game-over" if game_over else "paused" if paused else
                     "play" if active else "waiting")
            phase = {
                "level-up": "upgrade-menu", "treasure": "treasure-reward",
                "game-over": "game-over", "paused": "pause-menu",
                "play": "gameplay", "waiting": "unknown-screen",
            }[state]
            diagnostic_text = None
            if state == "waiting":
                if now - unknown_ocr_checked_at >= UNKNOWN_SCREEN_OCR_INTERVAL:
                    unknown_ocr_text = read_screen_text(frame)
                    unknown_ocr_checked_at = now
                diagnostic_text = unknown_ocr_text
                if "interestedinthis" in _label_key(unknown_ocr_text):
                    phase = "unlock-menu"
            else:
                unknown_ocr_checked_at = 0.0
                unknown_ocr_text = ""
            screen_reporter.publish(frame, phase, screen_text=diagnostic_text)
            fill = xp_bar_fill(frame)
            if chest_prompt_event:
                treasure_prompt_seen = True
                chest_detector.reset()
            chest_collected = chest_detector.update(frame, active=active)
            chest_collected = chest_collected or chest_prompt_event
            if chest_collected:
                run_chests += 1

            if chest_prompt and args.full_run:
                click_treasure_prompt(treasure_state)

            if active or level_up or game_over or chest_collected:
                observed_health = (
                    health_bar_fill(frame, player_x=args.player_x, player_y=args.player_y)
                    if active else None
                )
                new_kills = (kill_counter.sample(frame, now, final=game_over)
                             if active or game_over else 0)
                lamp_event = bool(gold_counter.sample(frame, now, final=game_over)
                                  if active or game_over else 0)
                reward = tracker.update(
                    fill, now - previous_time if action is not None else 0.0,
                    level_up_seen=level_up, chest_collected=chest_collected,
                    lamp_broken=lamp_event, health_fraction=observed_health, kills_gained=new_kills,
                    alive=not game_over,
                )
                action_name, danger = "-", 0.0
                total = reward.total
                run_xp += reward.xp_gain
                if action is not None:
                    choice, danger = action
                    action_name = "+".join(MOVES[choice])
                    total -= min(danger * 1.5, .5)
                    if learning_enabled:
                        counts[choice] += 1
                        means[choice] += (total - means[choice]) / counts[choice]
                        policy.update_learning_stats(counts, means)
                if dqn is not None:
                    terminal = game_over or level_up or chest_prompt_event
                    if active:
                        dqn_state = (dqn.reset(frame) if dqn_state is None
                                     else dqn.advance(frame))
                    elif terminal and pending_dqn is not None:
                        dqn_state = dqn.advance(frame)
                    if pending_dqn is not None and (active or terminal):
                        old_state, old_action = pending_dqn
                        loss = dqn.observe(old_state, old_action, total, dqn_state, terminal)
                        if loss is not None:
                            print(f"dqn-update:steps={dqn.steps},updates={dqn.updates},"
                                  f"loss={loss:.4f},epsilon={dqn.epsilon:.3f}", flush=True)
                        pending_dqn = None
                if active or state != last_state:
                    write_progress_report(
                        args.report_path, "progress", run_id=run_id,
                        completed_runs=completed_runs, elapsed_seconds=round(elapsed, 2),
                        state=state, xp_fill=round(fill, 4),
                        health_fill=observed_health, action=action_name,
                        kills=kill_counter.total, kill_reward=reward.kills,
                        xp_reward=reward.xp_gain, level_reward=reward.level_up,
                        chest_reward=reward.chest, lamp_reward=reward.lamp_break,
                        chicken_reward=reward.chicken, damage_penalty=reward.health_loss,
                        total_reward=total, dqn_steps=dqn.steps if dqn else None,
                        dqn_updates=dqn.updates if dqn else None,
                    )
                    print(
                        f"{elapsed:.2f},{fill:.3f},"
                        f"{observed_health if observed_health is not None else -1:.3f},"
                        f"{action_name},{danger:.4f},"
                        f"{kill_counter.total if kill_counter.total is not None else '-'},{reward.kills:.4f},"
                        f"{reward.xp_gain:.4f},{reward.level_up:.4f},{reward.chest:.4f},"
                        f"{reward.lamp_break:.4f},{reward.chicken:.4f},"
                        f"{reward.health_loss:.4f},{total:.4f},{state}",
                        flush=True,
                    )
            elif last_state != state:
                write_progress_report(
                    args.report_path, "progress", run_id=run_id,
                    completed_runs=completed_runs, elapsed_seconds=round(elapsed, 2),
                    state=state, xp_fill=round(fill, 4),
                    kills=kill_counter.total,
                    dqn_steps=dqn.steps if dqn else None,
                    dqn_updates=dqn.updates if dqn else None,
                )
                print(f"{elapsed:.2f},-,-,-,-,-,0,0,0,0,0,0,{state}", flush=True)
                tracker.reset()
            if not active:
                dqn_state = pending_dqn = None

            if evidence_run_dir and (elapsed >= next_evidence or state != last_state):
                output = evidence_run_dir / f"{elapsed:06.2f}-{state}.jpg"
                if not cv2.imwrite(str(output), frame):
                    raise RuntimeError(f"Could not save {output}")
                next_evidence = elapsed + args.evidence_interval

            session_deadline = start + args.seconds
            timed_out = not args.full_run and now >= session_deadline
            if timed_out and args.pause_at_end and level_up and not game_over:
                if end_choice_deadline is None:
                    end_choice_deadline = now + 60.0
                    print(
                        "session-time-limit: waiting up to 60 seconds for "
                        "the level-up choice, then pausing play",
                        flush=True,
                    )
                if now < end_choice_deadline:
                    action = None
                    if args.control:
                        bridge.release_all()
                else:
                    print("session-time-limit: left safely on level-up screen", flush=True)
                    break
            elif timed_out or game_over:
                if args.control:
                    bridge.release_all()
                if args.pause_at_end and active:
                    bridge.pause_game()
                if args.full_run and game_over:
                    run_result = {
                        "run_id": run_id, "seconds": round(elapsed, 2),
                        "game_seconds": read_game_timer_seconds(
                            frame, maximum=None if args.attach and completed_runs == 0 else elapsed + 2,
                        ),
                        "levels": run_levels, "chests": run_chests,
                        "kills": kill_counter.total,
                        "xp_reward": round(run_xp, 4), "picks": run_picks,
                        "result": "game_over",
                    }
                    if learning_enabled:
                        upgrades.record_run(run_result)
                    write_progress_report(
                        args.report_path, "run_completed", **run_result,
                        completed_runs=completed_runs + 1,
                        dqn_steps=dqn.steps if dqn else None,
                        dqn_updates=dqn.updates if dqn else None,
                    )
                    print(f"run-result:{json.dumps(run_result, sort_keys=True)}", flush=True)
                    completed_runs += 1
                    if args.max_runs and completed_runs >= args.max_runs:
                        print(f"run-limit: completed {completed_runs} runs", flush=True)
                        break
                    if dqn is not None and learning_enabled:
                        dqn.save()
                    print(f"run-restart: completed={completed_runs}; preparing next Gennaro / Inlaid Library run", flush=True)
                    return_to_title_after_death(
                        bridge, args.focus_timeout, reporter=screen_reporter,
                    )
                    run_id = uuid.uuid4().hex[:12]
                    write_progress_report(
                        args.report_path, "run_started", run_id=run_id,
                        completed_runs=completed_runs,
                    )
                    evidence_run_dir = args.evidence_dir / run_id
                    evidence_run_dir.mkdir(parents=True, exist_ok=True)
                    run_picks = []
                    run_levels = 0
                    run_chests = 0
                    run_xp = 0.0
                    tracker.reset()
                    kill_counter = KillCounter()
                    gold_counter = GoldCounter()
                    chest_detector = ChestPickupDetector(
                        player_x=args.player_x, player_y=args.player_y,
                    )
                    treasure_prompt_seen = False
                    treasure_click_at = float("-inf")
                    treasure_text_checked_at = 0.0
                    treasure_state_cache = None
                    over_text_checked_at = 0.0
                    over_state_cache = False
                    dqn_state = None
                    pending_dqn = None
                    action = None
                    last_state = None
                    next_evidence = 0.0
                    start_configured_run(
                        bridge, args.focus_timeout, reporter=screen_reporter, require_fresh=args.evaluate,
                    )
                    start = previous_time = time.monotonic()
                    continue
                break

            if level_up and args.full_run:
                bridge.release_all()
                run_levels += 1
                cards, raw = read_cards(frame)
                choice_index = upgrades.choose(cards, raw)
                chosen = cards[choice_index]
                print(
                    f"upgrade-cards:{json.dumps({'cards': cards, 'raw': raw}, ensure_ascii=False)}",
                    flush=True,
                )
                print(f"upgrade-choice:slot={choice_index + 1},name={chosen or 'OCR-unknown'}", flush=True)
                click_x, click_y = card_click_position(choice_index, frame)
                bridge.click_client(click_x, click_y)
                run_picks.append(chosen or f"Unknown card slot {choice_index + 1}")
                time.sleep(MENU_POST_CLICK_DELAY)
                verify = bridge.capture()
                if level_up_visible(verify):
                    print("upgrade-click: level-up panel remains; checking for a queued choice", flush=True)
                else:
                    print("upgrade-click: screen changed after selection", flush=True)
                action = None
                previous_time = time.monotonic()
                last_state = "level-up"
                continue

            if active:
                if dqn is not None and dqn_state is not None:
                    dqn_action = dqn.act(dqn_state, explore=learning_enabled)
                    choice = DQN_MOVE_INDICES[dqn_action]
                    action = (choice, 0.0)
                else:
                    scene = policy.observe(frame, level_up=False)
                    choice = scene.choice
                    action = (choice, scene.threat[choice])
                if args.control:
                    try:
                        bridge.set_movement(MOVES[choice])
                    except RuntimeError:
                        # Focus may change after the check at the top of the
                        # loop but before movement. Route that race through
                        # the same safe focus-loss handler on the next pass.
                        if bridge.is_foreground():
                            raise
                        bridge.release_all()
                        action = None
                        dqn_state = pending_dqn = None
                        tracker.reset()
                        previous_time = time.monotonic()
                        print(
                            "focus-lost: movement refused during action; "
                            "checking whether play needs pausing",
                            flush=True,
                        )
                        continue
                    if learning_enabled and dqn is not None and dqn_state is not None:
                        pending_dqn = (dqn_state, dqn_action)
            else:
                action = None
                if args.control and phase == "unlock-menu" and now - unlock_back_last_attempt >= 1.5:
                    bridge.click_client(.50, .80)
                    unlock_back_last_attempt = now
                    print("unlock-menu: clicked Back", flush=True)

            previous_time = now
            last_state = state
            period = args.action_seconds if active else .12
            deadline = end_choice_deadline if timed_out and end_choice_deadline is not None else session_deadline
            remaining = float("inf") if args.full_run else max(0, deadline - time.monotonic())
            if remaining:
                time.sleep(min(period, remaining))

    if learning_enabled:
        print(f"Action counts: {counts}. Statistics saved to {movement_stats}", flush=True)


if __name__ == "__main__":
    main()
