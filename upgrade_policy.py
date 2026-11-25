"""OCR and outcome-informed ranking for Vampire Survivors level-up cards."""

from __future__ import annotations

import difflib
import csv
import io
import json
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from xp_reward import six_choice_grid_visible, weapon_selection_grid_visible


ROOT = Path(__file__).parent
OCR_SCRIPT = ROOT / "windows_ocr.ps1"
STATS_PATH = ROOT / "data" / "upgrade_policy.json"
HISTORY_PATH = ROOT / "data" / "upgrade_history.jsonl"

# Static priors favor survival, reliable damage, and leveling. Run evidence
# updates these scores gradually; a few noisy games cannot overturn the priors.
BASE_VALUE = {
    "Laurel": 9.5, "Armor": 9.0, "Duplicator": 8.8, "Empty Tome": 8.6,
    "Pummarola": 8.4, "Garlic": 8.2, "Attractorb": 8.0, "Crown": 7.9,
    "Spinach": 7.8, "King Bible": 7.7, "Magic Wand": 7.6,
    "Fire Wand": 6.7, "Lightning Ring": 7.4, "Pentagram": 5.9,
    "Runetracer": 7.6, "Whip": 7.5, "Santa Javelin": 7.5,
    "Song of Mana": 7.7, "Gatti Amari": 6.4, "Shadow Pinion": 6.8,
    "Peachone": 7.0, "Ebony Wings": 7.0, "Phiera Der Tuphello": 7.2,
    "Eight The Sparrow": 7.2, "Vento Sacro": 7.3, "Glass Fandango": 7.3,
    "Bracelet": 6.8, "Unearthly Bolt": 7.6, "Ammo Appalate": 7.4,
    "Gaze of Gaea": 7.8, "Parma Aegis": 8.6,
    "Clock Lancet": 7.4, "Wings": 7.3, "Tiragisu": 7.2,
    "Hollow Heart": 7.2, "Spellbinder": 7.0, "Candelabrador": 7.0,
    "Bracer": 6.9, "Growth": 6.9, "Awake": 8.2, "Sarabande of Healing": 8.0,
    "Clover": 6.3, "Stone Mask": 5.8, "Silver Ring": 6.2,
    "Gold Ring": 4.0, "Metaglio Left": 6.8, "Metaglio Right": 3.8,
    "Skull O'Maniac": 2.8, "Torrona's Box": 4.3, "Curse": 2.0,
    "Greed": 3.8, "Move Speed": 5.8, "Recovery": 6.5,
    "Max Health": 6.4, "Might": 7.2, "Area": 6.9, "Amount": 8.7,
    "Cooldown": 8.5, "Duration": 6.7, "Magnet": 7.4,
    "Reroll": 5.5, "Skip": 3.0, "Banish": 5.7,
}

_ALIASES = {"Health": "Max Health", "Health Recovery": "Recovery",
            "Empty Heart": "Hollow Heart", "Move Speed": "Wings",
            "Magnet": "Attractorb"}
_CARD_ROWS = ((.245, .39), (.39, .535), (.535, .68), (.68, .825))
_CARD_Y = (.317, .462, .608, .753)


def _match_name(lines: list[str]) -> str | None:
    # First search exact known titles. OCR lines on one card are isolated from
    # the side panel, avoiding false matches on the player's current stats.
    text = re.sub(r"[^a-z0-9']+", " ", " ".join(lines).lower()).strip()
    for name in sorted(BASE_VALUE, key=len, reverse=True):
        key = re.sub(r"[^a-z0-9']+", " ", name.lower()).strip()
        if text == key or f" {key} " in f" {text} ":
            return _ALIASES.get(name, name)
    # Handle small OCR substitutions such as "Duplicat0r" without accepting
    # arbitrary description words as upgrade titles.
    for line in lines:
        cleaned = re.sub(r"[^a-z0-9']+", " ", line.lower()).strip()
        for word in cleaned.split():
            best = difflib.get_close_matches(word, [n.lower() for n in BASE_VALUE], n=1, cutoff=.82)
            if best and len(word) >= 5:
                original = next(name for name in BASE_VALUE if name.lower() == best[0])
                return _ALIASES.get(original, original)
    return None


def _match_grid_upgrade(lines: list[str]) -> str | None:
    """Map themed grid-card descriptions to the stat they improve."""
    text = re.sub(r"[^a-z0-9]+", " ", " ".join(lines).lower())
    for keyword, upgrade in (("cooldown", "Cooldown"), ("amount", "Amount"),
                             ("might", "Might"), ("duration", "Duration"),
                             ("area", "Area"), ("speed", "Move Speed")):
        if re.search(rf"\b{keyword}\b", text):
            return _ALIASES.get(upgrade, upgrade)
    return _match_name(lines)

def read_cards(frame: np.ndarray) -> tuple[list[str | None], list[str]]:
    """OCR the four-card list or the alternate two-column six-card grid."""
    h, w = frame.shape[:2]
    weapon_grid = weapon_selection_grid_visible(frame)
    tuna_grid = not weapon_grid and six_choice_grid_visible(frame)
    if weapon_grid:
        boxes = [(left, top, right, bottom)
                 for row, (top, bottom) in enumerate(
                     ((.20, .305), (.305, .41), (.41, .515), (.515, .62), (.62, .73)))
                 for column, (left, right) in enumerate(
                     ((.045, .275), (.28, .505), (.51, .735), (.74, .965)))
                 if not (row == 4 and column == 3)]
    elif tuna_grid:
        boxes = [(left, top, right, bottom)
                 for top, bottom in ((.24, .445), (.445, .63), (.63, .825))
                 for left, right in ((.055, .485), (.515, .945))]
    else:
        boxes = [(.34, top, .67, bottom) for top, bottom in _CARD_ROWS]
    found: list[str | None] = []
    raw: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vs-cards-") as folder:
        for index, (left, top, right, bottom) in enumerate(boxes):
            crop = frame[round(top * h):round(bottom * h),
                         round(left * w):round(right * w)]
            if os.name != "nt":
                text, lines = read_screen_ocr(crop)
                line_text = [str(item.get("text", "")) for item in lines]
                card_lines = line_text or text.splitlines()
                found.append(_match_grid_upgrade(card_lines) if tuna_grid else _match_name(card_lines))
                raw.append(" | ".join(line_text) if line_text else text)
                continue
            path = Path(folder) / f"card-{index}.png"
            if crop.size == 0 or not cv2.imwrite(str(path), crop):
                found.append(None)
                raw.append("")
                continue
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(OCR_SCRIPT), "-ImagePath", str(path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=20,
            )
            stdout = result.stdout or ""
            start, end = stdout.find("{"), stdout.rfind("}")
            if result.returncode or start < 0 or end < start:
                found.append(None)
                raw.append((result.stderr or stdout).strip()[-400:])
                continue
            try:
                parsed = json.loads(stdout[start:end + 1])
                lines = [str(item.get("text", "")) for item in parsed.get("lines", [])]
                found.append(_match_grid_upgrade(lines) if tuna_grid else _match_name(lines))
                raw.append(" | ".join(lines))
            except (ValueError, TypeError):
                found.append(None)
                raw.append("OCR JSON parse failed")
    return found, raw

def card_click_position(index: int, frame: np.ndarray) -> tuple[float, float]:
    """Return normalized click coordinates for a card in its visible layout."""
    if weapon_selection_grid_visible(frame):
        row, column = divmod(index, 4)
        return ((.16, .39, .62, .85)[column], (.25, .355, .46, .565, .67)[row])
    if six_choice_grid_visible(frame):
        row, column = divmod(index, 2)
        return ((.27, .73)[column], (.35, .54, .735)[row])
    return .50, _CARD_Y[index]


def read_screen_ocr(
    frame: np.ndarray, *, white_text: bool = False, page_mode: int | None = None,
    digits_only: bool = False, whitelist: str | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """Return full-screen OCR text and line boxes for menu navigation."""
    if os.name != "nt":
        if white_text:
            # Remove colorful menu artwork while preserving OCR coordinates.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 160), (179, 90, 255)))
        encoded_ok, encoded = cv2.imencode(".png", frame)
        if not encoded_ok:
            return "", []
        command = ["tesseract", "stdin", "stdout"]
        if page_mode is not None:
            command.extend(("--psm", str(page_mode)))
        if digits_only:
            whitelist = "0123456789"
        if whitelist is not None:
            command.extend(("-c", f"tessedit_char_whitelist={whitelist}"))
        try:
            result = subprocess.run(
                [*command, "tsv"], input=encoded.tobytes(),
                capture_output=True, timeout=12,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "", []
        if result.returncode:
            return "", []
        # Tesseract emits raw TSV. A quote recognized in a word is literal.
        rows = csv.DictReader(io.StringIO(result.stdout.decode("utf-8", errors="replace")),
                              delimiter="\t", quoting=csv.QUOTE_NONE)
        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in rows:
            word = row.get("text", "").strip()
            if not word:
                continue
            try:
                if float(row.get("conf", "-1")) < 0:
                    continue
                box = {"text": word, "x": int(row["left"]), "y": int(row["top"]),
                       "width": int(row["width"]), "height": int(row["height"])}
            except (KeyError, TypeError, ValueError):
                continue
            key = (row.get("block_num", "0"), row.get("par_num", "0"),
                   row.get("line_num", "0"))
            grouped.setdefault(key, []).append(box)
        lines: list[dict[str, object]] = []
        for words in grouped.values():
            left = min(int(word["x"]) for word in words)
            top = min(int(word["y"]) for word in words)
            right = max(int(word["x"]) + int(word["width"]) for word in words)
            bottom = max(int(word["y"]) + int(word["height"]) for word in words)
            lines.append({"text": " ".join(str(word["text"]) for word in words),
                          "x": left, "y": top, "width": right - left,
                          "height": bottom - top})
        return "\n".join(str(line["text"]) for line in lines), lines

    with tempfile.TemporaryDirectory(prefix="vs-screen-") as folder:
        path = Path(folder) / "screen.png"
        if not cv2.imwrite(str(path), frame):
            return "", []
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(OCR_SCRIPT), "-ImagePath", str(path)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20,
        )
        stdout = result.stdout or ""
        start, end = stdout.find("{"), stdout.rfind("}")
        if result.returncode or start < 0 or end < start:
            return "", []
        try:
            parsed = json.loads(stdout[start:end + 1])
            lines = parsed.get("lines", [])
            if isinstance(lines, dict):
                lines = [lines]
            elif not isinstance(lines, list):
                lines = []
            return str(parsed.get("text", "")), lines
        except (ValueError, TypeError):
            return "", []


def read_kill_count(frame: np.ndarray) -> int | None:
    """Read the digits immediately left of the skull in the gameplay HUD."""
    height, width = frame.shape[:2]
    crop = frame[round(.035 * height):round(.066 * height),
                 round(.72 * width):round(.793 * width)]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    digits = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 180), (179, 150, 255)))
    digits = cv2.resize(digits, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    digits = cv2.copyMakeBorder(digits, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    text, _ = read_screen_ocr(digits, page_mode=7, digits_only=True)
    text = text.strip()
    return int(text) if re.fullmatch(r"[0-9]{1,6}", text) else None


class KillCounter:
    """Sample the HUD at most once a second and count new kills once."""

    def __init__(self) -> None:
        self.total: int | None = None
        self._next_read = 0.0
        self._last_read = 0.0

    def sample(self, frame: np.ndarray, now: float, *, final: bool = False) -> int:
        if not final and now < self._next_read:
            return 0
        self._next_read = now + 1.0
        count = read_kill_count(frame)
        if count is None:
            return 0
        if self.total is None:
            self.total = count
            self._last_read = now
            return 0
        difference = count - self.total
        # Keep obvious OCR jumps out of replay. Readings may span missed samples.
        if not 0 <= difference <= max(16, 120 * (now - self._last_read)):
            return 0
        self.total = count
        self._last_read = now
        return difference


def read_gold_count(frame: np.ndarray) -> int | None:
    """Read the coin total immediately left of the gold icon in the HUD."""
    height, width = frame.shape[:2]
    crop = frame[round(.035 * height):round(.066 * height),
                 round(.875 * width):round(.927 * width)]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    digits = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 180), (179, 150, 255)))
    digits = cv2.resize(digits, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    digits = cv2.copyMakeBorder(digits, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    text, _ = read_screen_ocr(digits, page_mode=8, digits_only=True)
    text = text.strip()
    return int(text) if re.fullmatch(r"[0-9]{1,7}", text) else None


class GoldCounter:
    """Report one lamp/drop reward when the visible gold total increases."""

    def __init__(self) -> None:
        self.total: int | None = None
        self._next_read = 0.0
        self._last_read = 0.0

    def sample(self, frame: np.ndarray, now: float, *, final: bool = False) -> int:
        if not final and now < self._next_read:
            return 0
        self._next_read = now + 1.0
        count = read_gold_count(frame)
        if count is None:
            return 0
        if self.total is None:
            self.total = count
            self._last_read = now
            return 0
        difference = count - self.total
        # A single positive event represents collecting gold dropped by a
        # broken light source. Cap OCR jumps and score the event once.
        if not 0 < difference <= max(1000, 100 * (now - self._last_read)):
            return 0
        self.total = count
        self._last_read = now
        return 1


def read_game_over_heading(frame: np.ndarray) -> str:
    """Read the centered gold heading as a single line on Linux."""
    if os.name == "nt":
        return ""
    height, width = frame.shape[:2]
    heading = frame[round(.245 * height):round(.36 * height),
                    round(.34 * width):round(.66 * width)]
    if not heading.size:
        return ""
    hsv = cv2.cvtColor(heading, cv2.COLOR_BGR2HSV)
    gold = cv2.bitwise_not(cv2.inRange(hsv, (9, 150, 188), (38, 255, 255)))
    return read_screen_ocr(gold, page_mode=7)[0]


def read_game_timer_seconds(frame: np.ndarray, *, maximum: float | None = None) -> int | None:
    """Read the top-center gameplay clock from the Game Over screen."""
    height, width = frame.shape[:2]
    crop = frame[round(.05 * height):round(.082 * height),
                 round(.467 * width):round(.533 * width), :3]
    if not crop.size:
        return None
    timer = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(timer, cv2.COLOR_BGR2HSV)
    pale = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 144), (179, 119, 255)))
    gray = cv2.cvtColor(timer, cv2.COLOR_BGR2GRAY)
    _, contrast = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    readings = set()
    for image, page_mode in ((pale, 7), (contrast, 7), (gray, 8)):
        image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        text, _ = read_screen_ocr(image, page_mode=page_mode, whitelist="0123456789:")
        match = re.fullmatch(r"(\d{2}):([0-5]\d):?", text.strip())
        if match:
            seconds = 60 * int(match[1]) + int(match[2])
            if maximum is None or seconds <= maximum:
                readings.add(seconds)
    return readings.pop() if len(readings) == 1 else None


def read_screen_text(frame: np.ndarray) -> str:
    """Read all visible text on a game screen."""
    return read_screen_ocr(frame)[0]


class UpgradeLearner:
    """Rank visible cards and learn shared outcome associations across runs."""

    def __init__(self, policy_dir: Path | None = None) -> None:
        self.stats: dict[str, dict[str, float]] = {}
        self.stats_path = policy_dir / STATS_PATH.name if policy_dir else STATS_PATH
        self.history_path = policy_dir / HISTORY_PATH.name if policy_dir else HISTORY_PATH
        try:
            saved = json.loads(self.stats_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                self.stats = saved
        except (OSError, ValueError, TypeError):
            pass

    def choose(self, cards: list[str | None], raw: list[str] | None = None) -> int:
        raw = raw or []
        is_new = [bool(re.search(r"\bnew\b", raw[i], re.IGNORECASE))
                  if i < len(raw) else False for i in range(len(cards))]
        available = [i for i, name in enumerate(cards) if name]
        owned = [i for i in available if not is_new[i]]
        if owned:
            available = owned
        else:
            # OCR can read a card's New! badge even when it misses the title.
            # Prefer a readable non-new card over a new weapon in that case.
            readable_owned = [i for i in range(len(cards))
                              if i < len(raw) and raw[i].strip() and not is_new[i]]
            if readable_owned:
                return random.choice(readable_owned)
        if not available:
            # Never systematically favor the top card when OCR misses every
            # title. Selection remains unbiased until the OCR can identify it.
            return random.randrange(len(cards)) if cards else 0
        total = sum(float(item.get("count", 0)) for item in self.stats.values())
        def score(index: int) -> float:
            name = cards[index]
            assert name is not None
            item = self.stats.get(name, {})
            count = float(item.get("count", 0))
            mean = float(item.get("mean", .5))
            exploration = .45 * math.sqrt(math.log(total + 2) / (count + 1))
            learned = .70 * (mean - .5)
            return BASE_VALUE.get(name, 5.0) + learned + min(exploration, .7)
        best = max(score(index) for index in available)
        tied = [index for index in available if abs(score(index) - best) < 1e-9]
        return random.choice(tied)

    def record_run(self, run: dict[str, object]) -> None:
        reported_picks = sorted(set(str(x) for x in run.get("picks", []) if x))
        # Keep OCR misses in history for debugging, but don't train a shared
        # preference for a slot number whose actual upgrade title is unknown.
        picks = [name for name in reported_picks
                 if not name.lower().startswith("unknown card slot ")]
        seconds = float(run.get("seconds", 0))
        levels = float(run.get("levels", 0))
        chests = float(run.get("chests", 0))
        # A bounded terminal score includes survival, progression, and chest
        # collection. It is a comparative training signal, not a game rating.
        outcome = min(1.0, max(0.0, .62 * min(seconds / 900.0, 1.0)
                               + .23 * min(levels / 30.0, 1.0)
                               + .15 * min(chests / 10.0, 1.0)))
        for name in picks:
            item = self.stats.setdefault(name, {"count": 0, "mean": .5})
            n = float(item.get("count", 0)) + 1
            item["mean"] = float(item.get("mean", .5)) + (outcome - float(item.get("mean", .5))) / n
            item["count"] = n
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stats_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.stats, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.stats_path)
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**run, "outcome": outcome}, sort_keys=True) + "\n")


def card_click_y(index: int) -> float:
    return _CARD_Y[index]
