"""Simple screen-space threat avoidance for Vampire Survivors.

The policy scores nearby routes based on likely hostile sprites and XP gems.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MOVES: tuple[tuple[str, ...], ...] = (
    ("up",), ("down",), ("left",), ("right",),
    ("up", "left"), ("up", "right"), ("down", "left"), ("down", "right"),
)


def gameplay_hud_visible(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    border = frame[:max(5, round(height * .006)), width // 4:3 * width // 4]
    hsv = cv2.cvtColor(border, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (6, 80, 155), (42, 255, 255))
    return bool((gold > 0).mean(axis=1).max() > .80)


def pause_menu_visible(frame: np.ndarray) -> bool:
    """Recognize the large parchment-colored stats panel on the pause menu."""
    height, width = frame.shape[:2]
    panel = frame[
        round(0.17 * height):round(0.83 * height),
        round(0.30 * width):round(0.70 * width),
        :3,
    ]
    if panel.size == 0:
        return False
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    parchment = cv2.inRange(
        hsv, np.array((10, 0, 70)), np.array((40, 100, 220))
    )
    # The pause stats panel covers most of this crop; level-up cards do not.
    return float(np.count_nonzero(parchment) / parchment.size) >= 0.60


@dataclass(frozen=True)
class SceneEstimate:
    game_over: bool
    level_up: bool
    threat: tuple[float, ...]
    gems: tuple[float, ...]
    choice: int
    confidence: float


class ScreenSpaceDodge:
    """Choose a continuously moving lane from local pixel density.

    ``player_x`` and ``player_y`` are normalized screen coordinates. The
    current stage keeps the character near screen center; expose these as
    settings so they can be recalibrated if a later stage uses camera scroll.
    """

    def __init__(
        self,
        player_x: float = 0.50,
        player_y: float = 0.47,
        *,
        action_counts: tuple[int, ...] | list[int] | None = None,
        action_means: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        self.player_x = min(max(player_x, 0.05), 0.95)
        self.player_y = min(max(player_y, 0.12), 0.94)
        self._last_choice: int | None = None
        self._action_counts = [0] * len(MOVES)
        self._action_means = [0.0] * len(MOVES)
        self.update_learning_stats(action_counts, action_means)

    def update_learning_stats(
        self,
        counts: tuple[int, ...] | list[int] | None,
        means: tuple[float, ...] | list[float] | None,
    ) -> None:
        """Refresh the online reward prior without changing threat scoring."""
        if counts is not None and len(counts) == len(MOVES):
            self._action_counts = [max(0, int(value)) for value in counts]
        if means is not None and len(means) == len(MOVES):
            self._action_means = [float(value) for value in means]

    def _learned_preferences(self) -> tuple[float, ...]:
        scale = max(max((abs(value) for value in self._action_means), default=0.0), 0.02)
        return tuple(
            0.008 * (mean / scale) * (count / (count + 12.0))
            for mean, count in zip(self._action_means, self._action_counts)
        )

    @staticmethod
    def _threat_mask(hsv: np.ndarray) -> np.ndarray:
        # Bats and the Library's beige enemies both have dark outlines.
        dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 57))
        green = cv2.inRange(hsv, (36, 135, 40), (82, 255, 145))
        red = cv2.inRange(hsv, (0, 170, 60), (6, 255, 200))
        return dark | green | red

    @staticmethod
    def _gem_mask(hsv: np.ndarray) -> np.ndarray:
        cyan = cv2.inRange(hsv, (82, 145, 105), (115, 255, 255))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cyan)
        # Gems become tiny after the 320x180 reduction, so include 1-pixel
        # cores. Reject long, thin cyan weapon trails instead of relying on a
        # restrictive area ceiling that dropped many real pickups.
        widths = stats[:, cv2.CC_STAT_WIDTH]
        heights = stats[:, cv2.CC_STAT_HEIGHT]
        valid = ((stats[:, cv2.CC_STAT_AREA] >= 1)
                 & (stats[:, cv2.CC_STAT_AREA] <= 70)
                 & (widths <= 11) & (heights <= 13)
                 & (widths <= 4 * heights) & (heights <= 4 * widths))
        valid[0] = False
        return np.uint8(valid[labels]) * 255

    def _gem_attraction(self, mask: np.ndarray) -> tuple[float, ...]:
        """Score each route by how strongly it moves toward visible XP gems."""
        h, w = mask.shape
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        px, py = round(self.player_x * w), round(self.player_y * h)
        targets: list[tuple[float, float, float, float]] = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 1 or area > 70:
                continue
            gx, gy = centroids[label]
            dx, dy = gx - px, gy - py
            distance = float(np.hypot(dx, dy))
            if 3.0 <= distance <= 72.0:
                # Keep one large pile from overwhelming every other signal.
                strength = min(1.0, 0.45 + 0.12 * np.sqrt(area))
                targets.append((distance, dx / distance, dy / distance,
                                strength * np.exp(-distance / 42.0)))
        # Only the nearest cluster should drive the next short movement. Adding
        # every visible gem made every direction saturate on gem-heavy screens.
        targets.sort(key=lambda target: target[0])
        targets = targets[:12]
        total_weight = sum(target[3] for target in targets)

        attraction: list[float] = []
        for move in MOVES:
            dx = (1 if "right" in move else -1 if "left" in move else 0)
            dy = (1 if "down" in move else -1 if "up" in move else 0)
            norm = float(np.hypot(dx, dy)) or 1.0
            ux, uy = dx / norm, dy / norm
            value = 0.0
            for _, gx, gy, strength in targets:
                forward = max(0.0, gx * ux + gy * uy)
                cross = abs(gx * uy - gy * ux)
                value += strength * forward * np.exp(-cross / 0.18)
            attraction.append(value / max(total_weight, 1e-6))
        return tuple(attraction)

    @staticmethod
    def game_over_visible(frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        # The title screen may share gold text with this band, and Game Over
        # itself has a green Revive button. Keep this as a visual candidate;
        # train_agent confirms the exact "Game Over" text with OCR before it
        # exits so neither menu is misreported as a finished run.
        crop = frame_bgr[int(h * .26):int(h * .34), int(w * .36):int(w * .64)]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gold = cv2.inRange(hsv, (9, 150, 188), (38, 255, 255)) > 0
        text_sections = np.array_split(gold, 5, axis=1)
        return all(section.mean() > .025 for section in text_sections)

    def _lane_density(self, mask: np.ndarray, *, pickup: bool = False) -> tuple[float, ...]:
        h, w = mask.shape
        px, py = round(self.player_x * w), round(self.player_y * h)
        # Both axes use pixel distance. At this screen scale the character
        # can cover roughly 10 pixels per decision on the resized frame.
        radius = 10 if pickup else 7
        distances = (12, 22, 34) if pickup else (8, 15, 23)
        values: list[float] = []
        for move in MOVES:
            dx = (1 if "right" in move else -1 if "left" in move else 0)
            dy = (1 if "down" in move else -1 if "up" in move else 0)
            norm = 2 ** .5 if dx and dy else 1.0
            densities = []
            for distance in distances:
                tx = round(px + dx * distance / norm)
                ty = round(py + dy * distance / norm)
                x0, x1 = max(0, tx - radius), min(w, tx + radius + 1)
                y0, y1 = max(0, ty - radius), min(h, ty + radius + 1)
                area = mask[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else mask[:0, :0]
                densities.append(float(np.count_nonzero(area) / max(area.size, 1)))
            nearest, middle, farthest = densities
            values.append(.50 * nearest + .32 * middle + .18 * farthest)
        return tuple(values)

    def observe(self, frame_bgr: np.ndarray, level_up: bool = False) -> SceneEstimate:
        small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        threats = self._threat_mask(hsv)
        gems = self._gem_mask(hsv)
        threats[gems > 0] = 0
        player = (round(self.player_x * 320), round(self.player_y * 180))
        cv2.ellipse(threats, player, (5, 7), 0, 0, 360, 0, -1)
        for mask in (threats, gems):
            mask[:14, :] = 0
            mask[:, :17] = 0
            mask[:, 303:] = 0
        threat = self._lane_density(threats)
        gem_density = self._lane_density(gems, pickup=True)
        gem_attraction = self._gem_attraction(gems)
        safest = min(threat)
        candidates = [i for i, risk in enumerate(threat) if risk <= safest + .022]
        learned = self._learned_preferences()
        choice = min(candidates, key=lambda i: (
            threat[i] - .90 * gem_density[i] - .11 * gem_attraction[i]
            - learned[i]
            - (.005 if i == self._last_choice else 0),
            i not in (2, 3),  # Prefer a clear horizontal route on a tie.
        ))
        self._last_choice = choice
        confidence = min(1.0, max(0.0, (sorted(threat)[1] - sorted(threat)[0]) * 40))
        return SceneEstimate(
            game_over=(not level_up) and self.game_over_visible(frame_bgr),
            level_up=level_up,
            threat=threat,
            gems=gem_density,
            choice=choice,
            confidence=confidence,
        )
