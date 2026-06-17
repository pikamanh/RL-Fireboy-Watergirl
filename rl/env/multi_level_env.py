"""
Multi-level Gymnasium wrapper for FBWGEnv with three sampling strategies:
  - 'uniform'    : random level each episode
  - 'plr'        : Prioritized Level Replay (rank-based, focuses on hard levels)
  - 'curriculum' : unlock levels progressively as agent improves
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Literal, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .fbwg_env import FBWGEnv

SamplingMode = Literal["uniform", "plr", "curriculum"]


class PLRState:
    """
    Shared Prioritized Level Replay state.
    Pass the *same* instance to every MultiLevelEnv worker (DummyVecEnv only).

    Algorithm (PLR-lite):
        score[l]      = mean of last `window` episode returns for level l
        rank[l]       = 1 for the worst-scoring level (highest priority)
        score_w[l]    = 1 / rank[l]  (normalised)
        staleness_w[l]= steps_since_sampled[l] / sum(steps_since_sampled)
        weight[l]     = (1 - λ) * score_w[l] + λ * staleness_w[l]
    """

    def __init__(
        self,
        level_ids: List[int],
        mode: SamplingMode = "plr",
        window: int = 20,
        staleness_coef: float = 0.1,
        curriculum_threshold: float = -10.0,
        verbose: bool = True,
    ):
        self.level_ids = list(level_ids)
        self.mode = mode
        self.window = window
        self.staleness_coef = staleness_coef
        self.curriculum_threshold = curriculum_threshold
        self.verbose = verbose

        self._returns: dict[int, deque] = {l: deque(maxlen=window) for l in level_ids}
        self._staleness: dict[int, int] = {l: 0 for l in level_ids}
        self._episode_counts: dict[int, int] = {l: 0 for l in level_ids}

        # Curriculum: start with only the first (easiest) level
        self._available: list[int] = (
            [level_ids[0]] if mode == "curriculum" else list(level_ids)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, level_id: int, episode_return: float) -> None:
        self._returns[level_id].append(episode_return)
        self._episode_counts[level_id] += 1

    def sample(self) -> int:
        if self.mode == "uniform":
            return random.choice(self._available)
        elif self.mode == "plr":
            return self._plr_sample()
        else:
            return self._curriculum_sample()

    def summary(self) -> dict:
        return {
            l: {
                "mean_return": float(np.mean(self._returns[l])) if self._returns[l] else None,
                "episodes": self._episode_counts[l],
                "staleness": self._staleness[l],
                "available": l in self._available,
            }
            for l in self.level_ids
        }

    @property
    def available_levels(self) -> list[int]:
        return list(self._available)

    # ------------------------------------------------------------------
    # Sampling strategies
    # ------------------------------------------------------------------

    def _plr_sample(self) -> int:
        weights = self._plr_weights(self._available)
        level = random.choices(self._available, weights=weights, k=1)[0]
        self._update_staleness(level)
        return level

    def _curriculum_sample(self) -> int:
        self._maybe_unlock_next()
        level = random.choice(self._available)
        self._update_staleness(level)
        return level

    def _maybe_unlock_next(self) -> None:
        if len(self._available) >= len(self.level_ids):
            return
        filled = [self._returns[l] for l in self._available if self._returns[l]]
        if not filled:
            return
        avg_return = float(np.mean([np.mean(r) for r in filled]))
        if avg_return >= self.curriculum_threshold:
            next_level = self.level_ids[len(self._available)]
            self._available.append(next_level)
            if self.verbose:
                print(
                    f"\n[Curriculum] Unlocked level {next_level}! "
                    f"(avg_return={avg_return:.2f} >= {self.curriculum_threshold})"
                )

    def _plr_weights(self, levels: list[int]) -> list[float]:
        n = len(levels)
        scores = []
        for l in levels:
            scores.append(float(np.mean(self._returns[l])) if self._returns[l] else -1e9)

        # rank 1 = worst score = highest priority
        order = np.argsort(scores)
        ranks = np.empty(n, dtype=float)
        for pos, idx in enumerate(order):
            ranks[idx] = pos + 1
        score_w = 1.0 / ranks
        score_w /= score_w.sum()

        stale = np.array([self._staleness[l] for l in levels], dtype=float)
        stale_w = stale / (stale.sum() + 1e-8)

        w = (1 - self.staleness_coef) * score_w + self.staleness_coef * stale_w
        return w.tolist()

    def _update_staleness(self, sampled: int) -> None:
        for l in self.level_ids:
            self._staleness[l] += 1
        self._staleness[sampled] = 0


# ---------------------------------------------------------------------------


class MultiLevelEnv(gym.Env):
    """
    Gymnasium env that picks a random level on every reset.

    Observation space: FBWGEnv observation + one normalised level_id.
    Action space     : unchanged from FBWGEnv.

    Args:
        level_ids  : Levels to sample from, e.g. [1, 2, 3, 4, 5, 6].
        max_steps  : Episode length budget passed to FBWGEnv.
        plr_state  : Shared PLRState for sampling; uniform random if None.
        render_mode: Passed through to FBWGEnv.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        level_ids: List[int],
        max_steps: int = 3000,
        plr_state: Optional[PLRState] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert len(level_ids) >= 1, "Need at least one level"
        self.level_ids = list(level_ids)
        self.max_steps = max_steps
        self.plr_state = plr_state
        self.render_mode = render_mode

        self._level_min = min(level_ids)
        self._level_max = max(level_ids)

        # Bootstrap inner env to read action space
        self._current_level: int = level_ids[0]
        self._env: FBWGEnv = FBWGEnv(level_id=self._current_level, max_steps=max_steps)

        # Base obs + 1 normalised level feature.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._env.observation_space.shape[0] + 1,),
            dtype=np.float32,
        )
        self.action_space = self._env.action_space
        self._episode_return: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        # Report completed episode to PLR before switching level
        if self.plr_state is not None:
            self.plr_state.record(self._current_level, self._episode_return)
            self._current_level = self.plr_state.sample()
        else:
            self._current_level = random.choice(self.level_ids)

        self._env = FBWGEnv(level_id=self._current_level, max_steps=self.max_steps)
        obs, info = self._env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        info["level_id"] = self._current_level
        return self._aug(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._episode_return += reward
        info["level_id"] = self._current_level
        return self._aug(obs), reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------

    def _aug(self, obs: np.ndarray) -> np.ndarray:
        span = max(self._level_max - self._level_min, 1)
        level_norm = np.float32((self._current_level - self._level_min) / span)
        return np.append(obs, level_norm).astype(np.float32)
