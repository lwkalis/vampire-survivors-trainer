"""Visual Deep Q-Network policy for the eight movement actions.

The policy learns from live screen frames and the existing shaped XP/survival
reward. PyTorch is an optional dependency so the heuristic controller remains
available without it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise ImportError(
        "Deep learning mode requires PyTorch. Install it with "
        "`python -m pip install torch`."
    ) from exc


FRAME_SIZE = (84, 84)
STACK_SIZE = 4
ACTION_COUNT = 8
# Keep the DQN's output index aligned with scene_policy.MOVES:
# up, down, left, right, then the four diagonals.
DQN_MOVE_INDICES = tuple(range(ACTION_COUNT))


def frame_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Resize a game frame to a compact uint8 grayscale observation."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                  nn.Linear(512, ACTION_COUNT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class VisualDQN:
    """DQN with frame stacking, replay, a target network, and checkpoints."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str | None = None,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 0.0001,
        warmup: int = 128,
        train_every: int = 4,
        target_every: int = 2_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_steps: int = 30_000,
        seed: int = 7,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.gamma = gamma
        self.warmup = warmup
        self.train_every = train_every
        self.target_every = target_every
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_steps = epsilon_steps
        self.steps = 0
        self.updates = 0
        self.frames: deque[np.ndarray] = deque(maxlen=STACK_SIZE)
        self.replay: deque[Transition] = deque(maxlen=replay_capacity)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self._load_checkpoint()

    @property
    def epsilon(self) -> float:
        fraction = min(self.steps / max(self.epsilon_steps, 1), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        first = frame_to_gray(frame)
        self.frames.clear()
        self.frames.extend(first.copy() for _ in range(STACK_SIZE))
        return self._state()

    def _state(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0)

    def advance(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(frame_to_gray(frame))
        return self._state()

    def act(self, state: np.ndarray, *, explore: bool = True) -> int:
        if explore:
            self.steps += 1
            if random.random() < self.epsilon:
                return random.randrange(ACTION_COUNT)
        with torch.no_grad():
            tensor = torch.from_numpy(state).to(self.device).unsqueeze(0).float() / 255.0
            return int(self.online(tensor).argmax(dim=1).item())

    def observe(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> float | None:
        self.replay.append(Transition(state.copy(), action, float(reward),
                                      next_state.copy(), done))
        if len(self.replay) < max(self.batch_size, self.warmup):
            return None
        if self.steps % self.train_every:
            return None
        batch = random.sample(self.replay, self.batch_size)
        states = torch.from_numpy(np.stack([t.state for t in batch])).to(self.device).float() / 255.0
        actions = torch.tensor([t.action for t in batch], device=self.device).long()
        rewards = torch.tensor([t.reward for t in batch], device=self.device).float()
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(self.device).float() / 255.0
        dones = torch.tensor([t.done for t in batch], device=self.device).float()

        chosen_q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions[:, None]).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)
        loss = nn.functional.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu())

    def save(self) -> None:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint.with_suffix(".tmp")
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
        }, temporary)
        temporary.replace(self.checkpoint)

    def _load_checkpoint(self) -> None:
        if not self.checkpoint.exists():
            return
        saved = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.online.load_state_dict(saved["online"])
        self.target.load_state_dict(saved.get("target", saved["online"]))
        if "optimizer" in saved:
            self.optimizer.load_state_dict(saved["optimizer"])
        self.steps = int(saved.get("steps", 0))
        self.updates = int(saved.get("updates", 0))
