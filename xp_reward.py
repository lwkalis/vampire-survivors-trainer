"""XP bar perception and reward shaping for a local Vampire Survivors agent.

The game capture/keyboard bridge is intentionally separate. Feed this module
OpenCV BGR frames from the game window; the XP bar location and color thresholds
can be calibrated for the user's display settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_LEVEL_UP_TEMPLATE_PATH = Path(__file__).parent / "data" / "level_up_header.png"
_LEVEL_UP_TEMPLATE = (cv2.imread(str(_LEVEL_UP_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
                      if _LEVEL_UP_TEMPLATE_PATH.is_file() else None)


@dataclass(frozen=True)
class XPBarRegion:
    """Normalized XP bar crop plus observed full-bar violet-pixel ratio."""

    left: float = 0.05
    top: float = 0.006
    right: float = 0.95
    bottom: float = 0.03
    full_fill_color_ratio: float = 0.849


def xp_bar_fill(frame_bgr: np.ndarray, region: XPBarRegion = XPBarRegion()) -> float:
    """Estimate XP bar fill in [0, 1] by counting purple fill pixels.

    The full-bar reference comes from the paused level-up screen captured from
    this machine. Returns 0.0 if the crop is too small. Thresholds are
    configurable because color filters and game updates can change appearance.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
        raise ValueError("frame_bgr must be an HxWx3-or-more BGR image")

    height, width = frame_bgr.shape[:2]
    x0, x1 = round(region.left * width), round(region.right * width)
    y0, y1 = round(region.top * height), round(region.bottom * height)
    crop = frame_bgr[max(y0, 0):min(y1, height), max(x0, 0):min(x1, width), :3]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # The observed filled bar shifts between blue and violet. Exclude the
    # low-hue gold frame around it.
    fill_mask = cv2.inRange(hsv, np.array((90, 55, 55)), np.array((170, 255, 255)))
    raw_ratio = float(np.count_nonzero(fill_mask) / fill_mask.size)
    if region.full_fill_color_ratio <= 0:
        raise ValueError("full_fill_color_ratio must be positive")
    return min(raw_ratio / region.full_fill_color_ratio, 1.0)


def health_bar_fill(
    frame_bgr: np.ndarray,
    *,
    player_x: float = 0.50,
    player_y: float = 0.47,
    full_width: float = 0.0375,
) -> float | None:
    """Estimate the red player-health bar fill beneath the centered avatar.

    Returns ``None`` when the expected bar is not visible. The ROI is small
    and screen-relative. The full red span is 72 pixels in a 1920-pixel
    capture. Requiring saturated red excludes reddish terrain behind the bar.
    """
    height, width = frame_bgr.shape[:2]
    x0 = max(0, round((player_x - 0.03) * width))
    x1 = min(width, round((player_x + 0.03) * width))
    y0 = max(0, round((player_y + 0.034) * height))
    y1 = min(height, round((player_y + 0.046) * height))
    crop = frame_bgr[y0:y1, x0:x1, :3]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 200, 160), (8, 255, 255)) |
           cv2.inRange(hsv, (172, 200, 160), (179, 255, 255)))
    columns = (red > 0).sum(axis=0) >= max(3, round(crop.shape[0] * 0.25))
    # Close tiny gaps caused by pixel animation/compression before measuring.
    columns = cv2.morphologyEx(
        columns.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
        np.ones((1, 5), dtype=np.uint8),
    )[0] > 0
    edges = np.diff(np.pad(columns.astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    longest = int(np.max(ends - starts)) if len(starts) else 0
    if longest < max(5, round(width * 0.004)):
        return None
    return min(longest / max(round(width * full_width), 1), 1.0)


def six_choice_grid_visible(frame_bgr: np.ndarray) -> bool:
    """Detect a full-screen two-column, three-row upgrade choice layout."""
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    columns = ((.055, .485), (.515, .945))
    rows = ((.24, .445), (.445, .63), (.63, .825))
    detected = [[False, False] for _ in rows]
    for row_index, (top, bottom) in enumerate(rows):
        for column_index, (left, right) in enumerate(columns):
            crop = hsv[round(top * height):round(bottom * height),
                       round(left * width):round(right * width)]
            if crop.size == 0:
                continue
            # The six cards use broad, low-saturation slate interiors with
            # gold borders and colorful icons/text.
            central = crop[round(.30 * crop.shape[0]):round(.70 * crop.shape[0])]
            neutral = ((central[:, :, 1] < 85) & (central[:, :, 2] > 75)
                       & (central[:, :, 2] < 230))
            column_fill = (neutral.mean(axis=0) > .50)
            edges = np.diff(np.pad(column_fill.astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            longest = int(np.max(ends - starts)) if len(starts) else 0
            coverage = float(np.mean(neutral))
            # Each special choice is one wide card. The character menu has
            # several narrow portrait cards in these same regions, so require
            # a long continuous gray interior across each half of the screen.
            detected[row_index][column_index] = (
                coverage >= .34 and longest / max(crop.shape[1], 1) >= .60
            )
    rows_with_both_columns = sum(all(row) for row in detected)
    columns_present = all(any(row[column] for row in detected) for column in range(2))
    return rows_with_both_columns >= 2 and columns_present

def weapon_selection_grid_visible(frame_bgr: np.ndarray) -> bool:
    """Detect the wide four-column, five-row Weapon Selection panel."""
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    columns = ((.045, .275), (.28, .505), (.51, .735), (.74, .965))
    rows = ((.20, .305), (.305, .41), (.41, .515), (.515, .62), (.62, .73))
    detected = [[False] * len(columns) for _ in rows]
    for row_index, (top, bottom) in enumerate(rows):
        for column_index, (left, right) in enumerate(columns):
            crop = hsv[round(top * height):round(bottom * height),
                       round(left * width):round(right * width)]
            if crop.size == 0:
                continue
            central = crop[round(.20 * crop.shape[0]):round(.80 * crop.shape[0])]
            neutral = ((central[:, :, 1] < 95) & (central[:, :, 2] > 75)
                       & (central[:, :, 2] < 235))
            column_fill = neutral.mean(axis=0) > .48
            edges = np.diff(np.pad(column_fill.astype(np.int8), (1, 1)))
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            longest = int(np.max(ends - starts)) if len(starts) else 0
            detected[row_index][column_index] = (
                float(np.mean(neutral)) >= .27
                and longest / max(crop.shape[1], 1) >= .62
            )
    complete_rows = sum(sum(row) >= 3 for row in detected)
    complete_columns = all(sum(row[column] for row in detected) >= 4
                           for column in range(len(columns)))
    return complete_rows >= 4 and complete_columns


def _dominant_color_fraction(panel: np.ndarray, size: tuple[int, int]) -> float:
    small = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
    # Each channel has one bin per eight integer color values.
    counts = cv2.calcHist([small], [0, 1, 2], None, [32, 32, 32], [0, 256] * 3)
    return float(counts.max()) / (size[0] * size[1])


def level_up_visible(frame_bgr: np.ndarray) -> bool:
    """Detect the known centered Level Up heading, with a color fallback."""
    height, width = frame_bgr.shape[:2]
    if _LEVEL_UP_TEMPLATE is not None:
        scale = min(width / 2560.0, height / 1440.0)
        template = cv2.resize(
            _LEVEL_UP_TEMPLATE,
            (max(8, round(_LEVEL_UP_TEMPLATE.shape[1] * scale)),
             max(8, round(_LEVEL_UP_TEMPLATE.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        heading = frame_bgr[
            round(.11 * height):round(.23 * height),
            round(.38 * width):round(.63 * width),
        ]
        if (heading.shape[0] >= template.shape[0]
                and heading.shape[1] >= template.shape[1]):
            search = cv2.cvtColor(heading, cv2.COLOR_BGR2GRAY)
            score = float(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED).max())
            if score >= .72:
                return True

    # Handle alternate two-column upgrade layouts with a different title.
    if six_choice_grid_visible(frame_bgr) or weapon_selection_grid_visible(frame_bgr):
        return True

    # Fallback for screens with a different resolution/aspect ratio or a
    # missing template file.
    panel = frame_bgr[
        round(0.10 * height):round(0.90 * height),
        round(0.33 * width):round(0.67 * width),
        :3,
    ]
    if panel.size == 0:
        return False
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    gray_blue = cv2.inRange(
        hsv, np.array((90, 0, 40)), np.array((160, 100, 190))
    )
    coverage = float(np.count_nonzero(gray_blue) / gray_blue.size)
    # Library floors and newer stage artwork can exceed this color ratio in
    # ordinary play. A real card screen also has one broad, flat slate panel;
    # quantized dominant-color coverage separates that overlay from scenery.
    if coverage < 0.15 or _dominant_color_fraction(panel, (128, 180)) < 0.18:
        return False
    # Require at least two of the four gray card rows. A chest reward prompt
    # has the same large slate panel, but no level-up cards.
    card_rows = 0
    for top, bottom in ((.245, .39), (.39, .535), (.535, .68), (.68, .825)):
        row = cv2.cvtColor(
            frame_bgr[round(top * height):round(bottom * height),
                      round(.34 * width):round(.67 * width)], cv2.COLOR_BGR2HSV,
        )
        neutral = ((row[:, :, 1] < 50) & (row[:, :, 2] > 60) & (row[:, :, 2] < 210))
        column_fill = (neutral.mean(axis=0) > .50)
        edges = np.diff(np.pad(column_fill.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        longest = int(np.max(ends - starts)) if len(starts) else 0
        if (float(np.mean(neutral)) > 0.25
                and longest / max(row.shape[1], 1) > .60):
            card_rows += 1
    return card_rows >= 2


def treasure_overlay_candidate(frame_bgr: np.ndarray) -> bool:
    """Fast visual prefilter for a centered Treasure Found reward panel."""
    height, width = frame_bgr.shape[:2]
    panel = frame_bgr[
        round(.12 * height):round(.90 * height),
        round(.33 * width):round(.67 * width),
        :3,
    ]
    if panel.size == 0:
        return False
    return _dominant_color_fraction(panel, (96, 160)) > 0.20


@dataclass(frozen=True)
class RewardWeights:
    xp_fill: float = 1.0
    kill: float = 1.0
    level_up: float = 1.0
    chest: float = 1.0
    lamp_break: float = 1.0
    chicken: float = 1.0
    survival_per_second: float = 0.01
    death: float = 2.0
    health_loss: float = 18.0


@dataclass(frozen=True)
class RewardBreakdown:
    xp_gain: float
    level_up: float
    chest: float
    survival: float
    death: float
    health_loss: float = 0.0
    kills: float = 0.0
    lamp_break: float = 0.0
    chicken: float = 0.0

    @property
    def total(self) -> float:
        return (self.xp_gain + self.kills + self.level_up + self.chest + self.lamp_break
                + self.chicken + self.survival
                - self.health_loss - self.death)


@dataclass
class _ChestTrack:
    x: float
    y: float
    near_player: bool
    misses: int = 0


class ChestPickupDetector:
    """Reward a chest after its chest sprite disappears near the player.

    A small template from this game's visible chest is matched against a
    quarter-size screen image. Nearby presence alone is not rewarded; the
    chest must first be seen within pickup range and then disappear for two
    active frames.
    """

    def __init__(
        self,
        template_path: Path | None = None,
        *,
        player_x: float = 0.50,
        player_y: float = 0.47,
        match_threshold: float = 0.72,
        pickup_radius: float = 15.0,
    ) -> None:
        self.template_path = template_path or Path(__file__).parent / "data" / "chest_template.png"
        self.template = (cv2.imread(str(self.template_path), cv2.IMREAD_COLOR)
                         if self.template_path.is_file() else None)
        # Keep the trainer usable without redistributing a game-derived sprite.
        # Chest rewards from the opened modal are still detected separately.
        self.player_x = min(max(float(player_x), 0.05), 0.95)
        self.player_y = min(max(float(player_y), 0.12), 0.94)
        self.match_threshold = min(max(float(match_threshold), 0.0), 1.0)
        self.pickup_radius = max(float(pickup_radius), 1.0)
        self._tracks: list[_ChestTrack] = []

    def reset(self) -> None:
        """Forget sprite tracks after the pickup has been confirmed by its menu."""
        self._tracks.clear()

    def _candidates(self, frame_bgr: np.ndarray) -> list[tuple[float, float]]:
        if self.template is None:
            return []
        height, width = frame_bgr.shape[:2]
        scale = min(640 / width, 360 / height)
        scan_w, scan_h = max(1, round(width * scale)), max(1, round(height * scale))
        scan = cv2.resize(frame_bgr, (scan_w, scan_h), interpolation=cv2.INTER_AREA)
        template = cv2.resize(
            self.template,
            (max(2, round(self.template.shape[1] * scale)),
             max(2, round(self.template.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if template.shape[0] >= scan_h or template.shape[1] >= scan_w:
            return []
        scores = cv2.matchTemplate(scan, template, cv2.TM_CCOEFF_NORMED)
        # Ignore HUD icons; chests spawn in the playfield below the HUD.
        scores[:round(scan_h * .10), :] = -1.0
        candidates: list[tuple[float, float]] = []
        suppression_x = max(2, template.shape[1] // 2)
        suppression_y = max(2, template.shape[0] // 2)
        for _ in range(8):
            _, score, _, location = cv2.minMaxLoc(scores)
            if score < self.match_threshold:
                break
            x, y = location
            candidates.append((x + template.shape[1] / 2, y + template.shape[0] / 2))
            scores[max(0, y - suppression_y):y + suppression_y + 1,
                   max(0, x - suppression_x):x + suppression_x + 1] = -1.0
        return candidates

    def update(self, frame_bgr: np.ndarray, *, active: bool) -> bool:
        """Return true once a previously nearby chest is confirmed collected."""
        if not active or self.template is None:
            return False
        width, height = frame_bgr.shape[1], frame_bgr.shape[0]
        scale = min(640 / width, 360 / height)
        scan_w, scan_h = max(1, round(width * scale)), max(1, round(height * scale))
        px, py = self.player_x * scan_w, self.player_y * scan_h
        candidates = self._candidates(frame_bgr)
        used: set[int] = set()
        next_tracks: list[_ChestTrack] = []
        collected = False

        for track in self._tracks:
            nearest = min(
                ((i, (x - track.x) ** 2 + (y - track.y) ** 2)
                 for i, (x, y) in enumerate(candidates) if i not in used),
                key=lambda pair: pair[1], default=None,
            )
            if nearest is not None and nearest[1] <= 28.0 ** 2:
                i = nearest[0]
                used.add(i)
                x, y = candidates[i]
                track.x, track.y = x, y
                track.misses = 0
                track.near_player |= (x - px) ** 2 + (y - py) ** 2 <= self.pickup_radius ** 2
                next_tracks.append(track)
            else:
                track.misses += 1
                if track.near_player and track.misses >= 2:
                    collected = True
                elif track.misses < 5:
                    next_tracks.append(track)

        for i, (x, y) in enumerate(candidates):
            if i in used:
                continue
            near = (x - px) ** 2 + (y - py) ** 2 <= self.pickup_radius ** 2
            next_tracks.append(_ChestTrack(x, y, near))

        self._tracks = next_tracks
        return collected


class XPRewardTracker:
    """Turn XP-bar movement, level-ups, survival, and death into reward."""

    def __init__(self, weights: RewardWeights = RewardWeights()) -> None:
        self.weights = weights
        self._last_fill: float | None = None
        self._was_alive = True
        self._in_levelup = False
        self._health_reference: float | None = None

    def reset(self) -> None:
        self._last_fill = None
        self._was_alive = True
        self._in_levelup = False
        self._health_reference = None

    def update(
        self,
        fill: float,
        elapsed_seconds: float,
        *,
        level_up_seen: bool = False,
        chest_collected: bool = False,
        lamp_broken: bool = False,
        kills_gained: int = 0,
        health_fraction: float | None = None,
        alive: bool = True,
    ) -> RewardBreakdown:
        """Score one observation interval.

        ``level_up_seen`` can be supplied by a later HUD detector. A large XP
        bar reset is also treated as a level-up, accounting for progress that
        just rolled over into the next level.
        """
        fill = min(max(float(fill), 0.0), 1.0)
        dt = max(float(elapsed_seconds), 0.0)
        previous = self._last_fill

        if previous is None:
            xp_delta = 0.0
            inferred_level_up = False
        else:
            xp_delta = max(fill - previous, 0.0)
            inferred_level_up = not self._in_levelup and previous >= 0.80 and fill <= 0.25
            if inferred_level_up:
                xp_delta = 1.0 - previous + fill
        if self._in_levelup and level_up_seen:
            xp_delta = 0.0

        leveled = (level_up_seen and not self._in_levelup) or inferred_level_up
        died = self._was_alive and not alive
        health_loss = 0.0
        chicken = 0.0
        if health_fraction is not None:
            health_fraction = min(max(float(health_fraction), 0.0), 1.0)
            if self._health_reference is None:
                self._health_reference = health_fraction
            elif health_fraction > self._health_reference:
                # The player has no passive healing in the selected setup;
                # health restoration is the visible signal for a chicken pickup.
                if health_fraction - self._health_reference >= 0.025:
                    chicken = self.weights.chicken
                self._health_reference = health_fraction
            else:
                lost = self._health_reference - health_fraction
                if lost >= 0.025:
                    health_loss = min(lost * self.weights.health_loss, 4.0)
                    self._health_reference = health_fraction
        reward = RewardBreakdown(
            xp_gain=xp_delta * self.weights.xp_fill,
            kills=min(max(0, kills_gained) * self.weights.kill, 8.0),
            level_up=self.weights.level_up if leveled else 0.0,
            chest=self.weights.chest if chest_collected else 0.0,
            lamp_break=self.weights.lamp_break if lamp_broken else 0.0,
            chicken=chicken,
            survival=dt * self.weights.survival_per_second if alive else 0.0,
            death=self.weights.death if died else 0.0,
            health_loss=health_loss,
        )

        self._last_fill = None if level_up_seen else fill
        self._in_levelup = level_up_seen
        self._was_alive = alive
        return reward
