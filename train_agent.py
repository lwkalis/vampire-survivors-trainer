"""Run the local screen-space dodging baseline against Vampire Survivors."""

from __future__ import annotations

import argparse
import difflib
from datetime import datetime, timezone
import json
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
    read_screen_ocr, read_screen_text,
)


DATA_PATH = Path(__file__).parent / "data" / "vision_policy.json"
MENU_POLL_INTERVAL = .15
MENU_POST_CLICK_DELAY = .30
SCREEN_HEARTBEAT_INTERVAL = 1.0
SCREEN_PREVIEW_INTERVAL = 0.5
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


def load_stats() -> tuple[list[int], list[float]]:
    if not DATA_PATH.exists():
        return [0] * len(MOVES), [0.0] * len(MOVES)
    try:
        saved = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        counts = saved.get("counts", [])
        means = saved.get("means", [])
        if len(counts) == len(MOVES) and len(means) == len(MOVES):
            return [int(x) for x in counts], [float(x) for x in means]
    except (OSError, ValueError, TypeError):
        pass
    return [0] * len(MOVES), [0.0] * len(MOVES)


def save_stats(counts: list[int], means: list[float]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"counts": counts, "means": means, "actions": MOVES}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(DATA_PATH)


def _label_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def startup_screen_phase(screen: str) -> str:
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
        prefix_match = (
            wanted in {"quit", "options", "start"} and actual.startswith(wanted)
        ) or (wanted == "poe" and actual.startswith("poeratcho"))
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


def interested_page_visible(frame) -> bool:
    """Recognize the optional DLC prompt by its centered heading."""
    height, width = frame.shape[:2]
    title = frame[
        round(.105 * height):round(.22 * height),
        round(.28 * width):round(.72 * width),
    ]
    if not title.size:
        return False
    enlarged = cv2.resize(title, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text, _ = read_screen_ocr(enlarged, white_text=True, page_mode=7)
    normalized = _label_key(text)
    if "interested" in normalized and "this" in normalized:
        return True
    text, _ = read_screen_ocr(enlarged, page_mode=7)
    normalized = _label_key(text)
    return "interested" in normalized and "this" in normalized


def start_configured_run(
    bridge: VampireSurvivorsBridge, timeout: float, *, allow_active: bool = False,
    reporter: ScreenStatusReporter | None = None, first_launch: bool = False,
) -> None:
    """Navigate title → Poe → Mad Forest → gameplay, verifying each screen."""
    deadline = time.monotonic() + timeout
    title_start_clicked = False
    title_splash_clicked = False
    title_start_last_attempt = 0.0
    first_launch_advance_sent = False
    photosensitivity_attempts = 0
    run_start_clicked = False
    character_clicked = False
    character_confirmed = False
    library_unchecked_clicked = False
    mad_forest_clicked = False
    stage_clicked = False
    stage_start_clicked = False
    stage_start_last_attempt = 0.0
    interested_check_last_attempt = 0.0
    interested_back_last_attempt = 0.0
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
                    title_start_last_attempt = 0.0
                    run_start_clicked = False
                    character_clicked = False
                    character_confirmed = False
                    library_unchecked_clicked = False
                    mad_forest_clicked = False
                    stage_clicked = False
                    stage_start_clicked = False
                    last_message = None
                    continue
            if level_up_visible(frame):
                if reporter is not None:
                    reporter.publish(frame, "upgrade-menu")
                print("startup: open upgrade choice found; resuming the active run", flush=True)
                return
            if gameplay_hud_visible(frame):
                if reporter is not None:
                    reporter.publish(frame, "gameplay")
                if allow_active or run_start_clicked:
                    print("ready: configured Poe / Mad Forest run is active", flush=True)
                    return
                raise RuntimeError(
                    "Gameplay is already active. Start the new run from the title menu "
                    "or use --attach to continue this run."
                )
            now = time.monotonic()
            if now - interested_check_last_attempt >= 0.75:
                interested_check_last_attempt = now
                if interested_page_visible(frame):
                    if reporter is not None:
                        reporter.publish(frame, "promo-page", screen_text="Interested in this?")
                    if now - interested_back_last_attempt >= 1.0:
                        bridge.release_all()
                        bridge.click_client(.50, .80)
                        interested_back_last_attempt = now
                        message = "promo-page: clicked Back"
                    else:
                        message = "promo-page: waiting for Back to close"
                    if message != last_message:
                        print(message, flush=True)
                        last_message = message
                    time.sleep(MENU_POLL_INTERVAL)
                    continue
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
            if "photosensitivity" in screen and "warning" in screen:
                # The first-launch disclaimer can require an extra input after
                # the initial acknowledgment. Click its lower-center button,
                # then tap Enter once if the warning remains visible.
                if photosensitivity_attempts < 2:
                    photosensitivity_attempts += 1
                    if photosensitivity_attempts == 1:
                        bridge.click_client(.50, .82)
                        message = "startup: acknowledged the photosensitivity warning"
                    else:
                        bridge.press_key("enter")
                        message = "startup: sent extra photosensitivity input"
                else:
                    message = "startup: waiting for photosensitivity warning to clear"
            elif (
                first_launch and not first_launch_advance_sent
                and not any(label in screen for label in (
                    "characterselection", "stageselection", "firstsurvivaton",
                    "quickstart", "vampiresurvivors", "gameover",
                ))
            ):
                # OCR can miss the first-launch disclaimer entirely. A single
                # Enter advances it; inspect the new screen before acting again.
                bridge.press_key("enter")
                first_launch_advance_sent = True
                message = "startup: sent first-launch continue input"
            elif "gameover" in screen:
                print("startup: Game Over detected in OCR; returning to title", flush=True)
                remaining = max(.1, deadline - time.monotonic())
                return_to_title_after_death(bridge, remaining, reporter=reporter)
                title_start_clicked = False
                title_splash_clicked = False
                title_start_last_attempt = 0.0
                run_start_clicked = False
                character_clicked = False
                character_confirmed = False
                library_unchecked_clicked = False
                mad_forest_clicked = False
                stage_clicked = False
                stage_start_clicked = False
                last_message = None
                continue
            elif "characterselection" in screen:
                if not character_clicked:
                    if click_ocr_label(bridge, frame, "Poe", ocr=(screen_text, screen_lines)):
                        character_clicked = True
                        message = "selection: chose Poe"
                    else:
                        message = "selection: waiting for the Poe card"
                elif not character_confirmed:
                    if not click_ocr_label(bridge, frame, "Confirm", (.824, .93), (screen_text, screen_lines)):
                        bridge.click_client(.824, .93)
                    character_confirmed = True
                    message = "selection: confirmed Poe"
                elif not stage_checkbox_checked(frame, "Mad Forest"):
                    if not mad_forest_clicked:
                        bridge.click_client(.752, .228)
                        mad_forest_clicked = True
                        message = "selection: clicked Mad Forest checkbox"
                    else:
                        message = "selection: waiting for Mad Forest to be selected"
                elif stage_checkbox_checked(frame, "Inlaid Library"):
                    if not library_unchecked_clicked:
                        bridge.click_client(.752, .260)
                        library_unchecked_clicked = True
                        message = "selection: unchecked Inlaid Library"
                    else:
                        message = "selection: waiting for Inlaid Library to clear"
                else:
                    start_button_clicked = click_ocr_label(
                        bridge, frame, "Start", (.824, .93), (screen_text, screen_lines),
                    )
                    if not start_button_clicked:
                        bridge.click_client(.824, .93)
                    run_start_clicked = True
                    message = "selection: starting Poe in Mad Forest"
            elif "stageselection" in screen:
                if not stage_clicked:
                    if click_ocr_label(bridge, frame, "Mad Forest", ocr=(screen_text, screen_lines)):
                        stage_clicked = True
                        message = "selection: chose Mad Forest"
                    else:
                        # Mad Forest is the first stage card. Its title is small
                        # and OCR often misses it on this screen.
                        bridge.click_client(.365, .255)
                        stage_clicked = True
                        message = "selection: clicked the Mad Forest card"
                elif not stage_start_clicked:
                    # The Start label is sometimes omitted by Windows OCR on
                    # this busy stage screen; use the calibrated button point.
                    bridge.click_client(.824, .93)
                    stage_start_clicked = True
                    stage_start_last_attempt = time.monotonic()
                    run_start_clicked = True
                    message = "selection: starting Mad Forest"
                elif time.monotonic() - stage_start_last_attempt >= 2.0:
                    # Retry if the menu did not accept the first click.
                    bridge.click_client(.824, .93)
                    stage_start_last_attempt = time.monotonic()
                    message = "selection: retrying Mad Forest Start"
                else:
                    message = "selection: waiting for Mad Forest to start"
            elif (
                title_splash_clicked
                or "firstsurvivaton" in screen
                or "quickstart" in screen
                or "vampiresurvivors" in screen
            ):
                # The title tagline can appear on the menu while OCR misses
                # the visible START label. Retry the calibrated button until
                # the character/stage selection screen is recognized.
                now = time.monotonic()
                if not title_start_clicked or now - title_start_last_attempt >= 1.5:
                    retrying = title_start_clicked
                    if not click_ocr_label(
                        bridge, frame, "Start", (.50, .65), (screen_text, screen_lines),
                    ):
                        bridge.click_client(.50, .65)
                    title_start_clicked = True
                    title_splash_clicked = True
                    title_start_last_attempt = now
                    message = (
                        "title-menu: retrying Start" if retrying
                        else "title-menu: clicked Start"
                    )
                else:
                    message = "menu-check: waiting for title menu response"
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
    interested_check_last_attempt = 0.0
    interested_back_last_attempt = 0.0
    last_message = None
    while time.monotonic() < deadline:
        if not bridge.is_foreground():
            message = "restart: focus the game; waiting without input"
            if reporter is not None:
                reporter.publish(None, "focus-required")
        else:
            frame = bridge.capture()
            now = time.monotonic()
            if now - interested_check_last_attempt >= 0.75:
                interested_check_last_attempt = now
                if interested_page_visible(frame):
                    if reporter is not None:
                        reporter.publish(frame, "promo-page", screen_text="Interested in this?")
                    if now - interested_back_last_attempt >= 1.0:
                        bridge.release_all()
                        bridge.click_client(.50, .80)
                        interested_back_last_attempt = now
                        message = "promo-page: clicked Back"
                    else:
                        message = "promo-page: waiting for Back to close"
                    if message != last_message:
                        print(message, flush=True)
                        last_message = message
                    time.sleep(MENU_POLL_INTERVAL)
                    continue
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
                        help="select Poe and Mad Forest, learn, and restart after death")
    parser.add_argument("--attach", action="store_true",
                        help="with --full-run, accept an already active run as the first run")
    parser.add_argument("--deep-learning", action="store_true",
                        help="use a visual DQN movement policy (requires PyTorch)")
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
    if args.full_run:
        args.control = True
        if args.evidence_dir is None:
            args.evidence_dir = Path(__file__).parent / "run-evidence" / uuid.uuid4().hex[:12]
        if args.evidence_interval == 1.5:
            args.evidence_interval = 6.0
    if args.attach and not args.full_run:
        parser.error("--attach requires --full-run")
    if (args.seconds <= 0 or args.action_seconds <= 0 or args.max_runs < 0
            or args.evidence_interval <= 0 or args.focus_timeout <= 0):
        parser.error("session, action, evidence, and focus durations must be positive; max-runs cannot be negative")
    if args.pause_at_end and not args.control:
        parser.error("--pause-at-end requires --control")
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
        dqn = VisualDQN(Path(__file__).parent / "data" / "visual_dqn_inlaid_library_8way.pt")
        print(f"deep-learning: visual DQN on {dqn.device}; epsilon={dqn.epsilon:.3f}; "
              f"steps={dqn.steps}; updates={dqn.updates}", flush=True)
    upgrades = UpgradeLearner()
    run_id = uuid.uuid4().hex[:12]
    completed_runs = 0
    write_progress_report(
        args.report_path, "session_started", run_id=run_id,
        deep_learning=args.deep_learning, max_runs=args.max_runs,
    )
    evidence_run_dir = (args.evidence_dir / run_id
                        if args.evidence_dir and args.full_run else args.evidence_dir)
    if evidence_run_dir:
        evidence_run_dir.mkdir(parents=True, exist_ok=True)
    run_picks: list[str] = []
    run_levels = 0
    run_chests = 0
    run_xp = 0.0
    counts, means = load_stats()
    policy = ScreenSpaceDodge(
        args.player_x, args.player_y,
        action_counts=counts, action_means=means,
    )
    tracker = XPRewardTracker()
    kill_counter = KillCounter()
    gold_counter = GoldCounter()
    chest_detector = ChestPickupDetector(player_x=args.player_x, player_y=args.player_y)
    treasure_open_latched = False
    treasure_text_checked_at = 0.0
    treasure_state_cache: str | None = None
    level_text_checked_at = 0.0
    level_state_cache = False
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
        elif treasure_open_latched:
            treasure_state_cache = "animation"
        else:
            treasure_state_cache = None
        return treasure_state_cache

    if args.control and args.full_run:
        bridge.activate()  # Best effort; foreground activation may need user help.
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
            reporter=screen_reporter, first_launch=True,
        )
    elif args.control:
        bridge.activate()  # Best effort; Windows may require a user click.
        ready_deadline = time.monotonic() + args.focus_timeout
        waiting_message = None
        quick_start_clicked = False
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
                menu_text = read_screen_text(ready_frame).lower() if args.full_run else ""
                ready_level_up = confirmed_level_up(ready_frame)
                ready_treasure_state = (None if ready_level_up
                                        else treasure_screen_state(ready_frame))
                if ready_treasure_state == "done":
                    print("treasure-reward: clicking Done", flush=True)
                    bridge.click_client(.50, .805)
                    if not treasure_open_latched:
                        run_chests += 1
                    treasure_open_latched = False
                    treasure_state_cache = None
                    time.sleep(.35)
                    verify_done = bridge.capture()
                    verify_text = (read_screen_text(verify_done).lower()
                                   if treasure_overlay_candidate(verify_done) else "")
                    if "done" in verify_text:
                        treasure_text_checked_at = 0.0
                        treasure_state_cache = "done"
                        message = "treasure-done: verifying the confirmation click"
                    else:
                        treasure_text_checked_at = 0.0
                        message = "treasure-reward-claimed: waiting for gameplay"
                elif ready_treasure_state in {"open", "animation"}:
                    if ready_treasure_state == "open" and not treasure_open_latched:
                        print("treasure-found: opening the chest reward", flush=True)
                        bridge.click_client(.50, .805)
                        treasure_open_latched = True
                        run_chests += 1
                        treasure_state_cache = "animation"
                        time.sleep(.45)
                    message = "treasure-opened: waiting for reward reveal"
                elif (confirmed_game_over(ready_frame)
                        and "quick start" not in menu_text):
                    raise RuntimeError(
                        "The visible run is at Game Over. Start or revive a run, "
                        "then launch the controller again."
                    )
                elif ready_level_up:
                    if args.full_run:
                        bridge.release_all()
                        cards, raw = read_cards(ready_frame)
                        choice_index = upgrades.choose(cards)
                        chosen = cards[choice_index]
                        print(f"upgrade-cards:{json.dumps({'cards': cards, 'raw': raw}, ensure_ascii=False)}", flush=True)
                        print(f"upgrade-choice:slot={choice_index + 1},name={chosen or 'OCR-unknown'}", flush=True)
                        click_x, click_y = card_click_position(choice_index, ready_frame)
                        bridge.click_client(click_x, click_y)
                        run_picks.append(chosen or f"Unknown card slot {choice_index + 1}")
                        run_levels += 1
                        time.sleep(MENU_POST_CLICK_DELAY)
                        level_text_checked_at = 0.0
                        level_state_cache = False
                        message = "upgrade-selected: waiting for active gameplay"
                    else:
                        message = "level-up: choose a card; waiting without movement"
                elif pause_menu_visible(ready_frame):
                    message = "paused: resume the game; waiting without movement"
                elif gameplay_hud_visible(ready_frame):
                    if args.full_run and not (quick_start_clicked or args.attach):
                        raise RuntimeError(
                            "Full-run mode requires the title menu so it can start a fresh run; "
                            "the screen is already in active gameplay."
                        )
                    if args.full_run:
                        label = "attached-run" if args.attach else "new-run"
                        print(f"ready: {label} HUD confirmed", flush=True)
                    print("ready: game focused and active; controller starting", flush=True)
                    break
                elif args.full_run:
                    if args.attach:
                        message = "attach-mode: choose the same character and start the run; waiting for gameplay"
                    elif "quick start" in menu_text and not quick_start_clicked:
                        print("title-menu: OCR verified Quick Start; clicking it", flush=True)
                        bridge.click_client(.65, .65)
                        quick_start_clicked = True
                        message = "starting: waiting for the new-run HUD"
                    elif quick_start_clicked:
                        message = "starting: waiting for the new-run HUD"
                    else:
                        message = "menu-check: Quick Start not confirmed; no click sent"
                else:
                    message = "waiting: looking for active gameplay; no movement sent"
                if ready_level_up:
                    ready_phase = "upgrade-menu"
                elif ready_treasure_state is not None:
                    ready_phase = "treasure-reward"
                elif confirmed_game_over(ready_frame):
                    ready_phase = "game-over"
                elif pause_menu_visible(ready_frame):
                    ready_phase = "pause-menu"
                elif gameplay_hud_visible(ready_frame):
                    ready_phase = "gameplay"
                elif "quick start" in menu_text:
                    ready_phase = "title-menu"
                else:
                    ready_phase = "unknown-screen"
                screen_reporter.publish(
                    ready_frame, ready_phase,
                    screen_text=menu_text if menu_text else None,
                )
            if message != waiting_message:
                print(message, flush=True)
                waiting_message = message
            time.sleep(MENU_POLL_INTERVAL)

    start = previous_time = time.monotonic()
    next_evidence = 0.0
    unknown_ocr_checked_at = 0.0
    unknown_ocr_text = ""
    action = None
    last_state = None
    focus_lost_since: float | None = None
    end_choice_deadline: float | None = None
    print("elapsed_s,xp_fill,health_fill,action,danger,kills,kill_reward,xp_reward,level_reward,chest_reward,lamp_reward,chicken_reward,damage_penalty,total,state", flush=True)
    try:
        while True:
            if args.control and not bridge.is_foreground():
                screen_reporter.publish(None, "focus-required")
                bridge.release_all()
                action = None
                if focus_lost_since is None:
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
            if not treasure:
                treasure_open_latched = False
            chest_prompt_event = (treasure_state in {"open", "done"}
                                  and not treasure_open_latched)
            game_over = not (level_up or treasure) and confirmed_game_over(frame)
            paused = not (level_up or treasure or game_over) and pause_menu_visible(frame)
            active = not (level_up or treasure or game_over or paused) and gameplay_hud_visible(frame)
            state = ("level-up" if level_up else "treasure" if treasure else
                     "game-over" if game_over else "paused" if paused else
                     "play" if active else "waiting")
            phase = {
                "level-up": "upgrade-menu", "treasure": "treasure-reward",
                "game-over": "game-over", "paused": "pause-menu",
                "play": "gameplay", "waiting": "unknown-screen",
            }[state]
            diagnostic_text = None
            if state == "waiting" and now - unknown_ocr_checked_at >= UNKNOWN_SCREEN_OCR_INTERVAL:
                unknown_ocr_text = read_screen_text(frame)
                unknown_ocr_checked_at = now
                diagnostic_text = unknown_ocr_text
            screen_reporter.publish(frame, phase, screen_text=diagnostic_text)
            fill = xp_bar_fill(frame)
            chest_collected = chest_detector.update(frame, active=active)
            chest_collected = chest_collected or chest_prompt_event
            if chest_collected:
                run_chests += 1

            if treasure_state == "open" and not treasure_open_latched and args.full_run:
                bridge.release_all()
                print("treasure-found: opening the chest reward", flush=True)
                bridge.click_client(.50, .805)
                treasure_open_latched = True
                treasure_state_cache = "animation"
            elif treasure_state == "done" and args.full_run:
                bridge.release_all()
                print("treasure-reward: clicking Done", flush=True)
                bridge.click_client(.50, .805)
                treasure_open_latched = False
                treasure_state_cache = None

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
                    if args.control:
                        counts[choice] += 1
                        means[choice] += (total - means[choice]) / counts[choice]
                        policy.update_learning_stats(counts, means)
                if dqn is not None and (active or level_up or game_over):
                    if active:
                        dqn_state = (dqn.reset(frame) if dqn_state is None
                                     else dqn.advance(frame))
                    elif dqn_state is not None:
                        dqn_state = dqn.advance(frame)
                    if pending_dqn is not None and dqn_state is not None:
                        old_state, old_action = pending_dqn
                        loss = dqn.observe(old_state, old_action, total, dqn_state,
                                           game_over or level_up)
                        if loss is not None:
                            print(f"dqn-update:steps={dqn.steps},updates={dqn.updates},"
                                  f"loss={loss:.4f},epsilon={dqn.epsilon:.3f}", flush=True)
                        pending_dqn = None
                    if level_up or game_over:
                        dqn_state = None
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

            if evidence_run_dir and (elapsed >= next_evidence or state != last_state):
                output = evidence_run_dir / f"{elapsed:06.2f}-{state}.jpg"
                if not cv2.imwrite(str(output), frame):
                    raise RuntimeError(f"Could not save {output}")
                next_evidence = elapsed + args.evidence_interval

            timed_out = (not args.full_run) and elapsed >= args.seconds
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
                        "levels": run_levels, "chests": run_chests,
                        "kills": kill_counter.total,
                        "xp_reward": round(run_xp, 4), "picks": run_picks,
                        "result": "game_over",
                    }
                    upgrades.record_run(run_result)
                    write_progress_report(
                        args.report_path, "run_completed", **run_result,
                        completed_runs=completed_runs + 1,
                        dqn_steps=dqn.steps if dqn else None,
                        dqn_updates=dqn.updates if dqn else None,
                    )
                    print(f"run-result:{json.dumps(run_result, sort_keys=True)}", flush=True)
                    completed_runs += 1
                    if dqn is not None:
                        dqn.save()
                    if args.max_runs and completed_runs >= args.max_runs:
                        print(f"run-limit: completed {completed_runs} runs", flush=True)
                        break
                    print(f"run-restart: completed={completed_runs}; preparing next Poe / Mad Forest run", flush=True)
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
                    treasure_open_latched = False
                    treasure_text_checked_at = 0.0
                    treasure_state_cache = None
                    level_text_checked_at = 0.0
                    level_state_cache = False
                    over_text_checked_at = 0.0
                    over_state_cache = False
                    dqn_state = None
                    pending_dqn = None
                    action = None
                    last_state = None
                    next_evidence = 0.0
                    start_configured_run(
                        bridge, args.focus_timeout, reporter=screen_reporter,
                    )
                    start = previous_time = time.monotonic()
                    continue
                break

            if level_up and args.full_run:
                bridge.release_all()
                run_levels += 1
                cards, raw = read_cards(frame)
                choice_index = upgrades.choose(cards)
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
                level_text_checked_at = 0.0
                level_state_cache = False
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
                    dqn_action = dqn.act(dqn_state)
                    choice = DQN_MOVE_INDICES[dqn_action]
                    action = (choice, 0.0)
                    pending_dqn = (dqn_state.copy(), dqn_action)
                else:
                    scene = policy.observe(frame, level_up=False)
                    action = (scene.choice, scene.threat[scene.choice])
                if args.control:
                    try:
                        bridge.set_movement(MOVES[choice])
                    except RuntimeError:
                        # Focus may change after the check at the top of the
                        # loop but before SendInput. Route that race through
                        # the same safe focus-loss handler on the next pass.
                        if bridge.is_foreground():
                            raise
                        bridge.release_all()
                        action = None
                        focus_lost_since = time.monotonic()
                        print(
                            "focus-lost: movement refused during action; "
                            "checking whether play needs pausing",
                            flush=True,
                        )
            else:
                action = None
                if args.control:
                    bridge.release_all()

            previous_time = now
            last_state = state
            period = args.action_seconds if active else .12
            remaining = float("inf") if args.full_run else max(0, args.seconds - elapsed)
            time.sleep(min(period, remaining))
    finally:
        write_progress_report(
            args.report_path, "session_stopped", completed_runs=completed_runs,
            dqn_steps=dqn.steps if dqn else None,
            dqn_updates=dqn.updates if dqn else None,
        )
        if args.control:
            bridge.release_all()
            save_stats(counts, means)
        if dqn is not None:
            dqn.save()
            print(f"deep-learning-saved: steps={dqn.steps}, updates={dqn.updates}, "
                  f"checkpoint={dqn.checkpoint}", flush=True)

    if args.control:
        print(f"Action counts: {counts}. Statistics saved to {DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
