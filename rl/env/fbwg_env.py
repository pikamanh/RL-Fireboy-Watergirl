"""
Gymnasium environment for Fireboy and Watergirl.

Observation (14 floats, all in [0, 1]):
    fb_x, fb_y, fb_vx, fb_vy,
    wg_x, wg_y, wg_vx, wg_vy,
    fire_door_x, fire_door_y,
    water_door_x, water_door_y,
    fb_on_ground, wg_on_ground

Action (MultiDiscrete [4, 4]):
    [fireboy_action, watergirl_action]
    0 = idle, 1 = left, 2 = right, 3 = jump
"""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .physics import (
    PlayerState, step_player,
    CANVAS_W, CANVAS_H, MOVE_VX, JUMP_VY,
    N_ACTIONS,
)
from .level_loader import load_level_tiles, load_level_data

# Door geometry (from door.js)
DOOR_HB_DX = 20
DOOR_HB_DY = 20
DOOR_HB_W  = 60
DOOR_HB_H  = 88

# Normalisation helpers
MAX_VX = MOVE_VX + 0.5      # small buffer beyond max horizontal speed
MAX_VY_UP   = abs(JUMP_VY) + 1.0
MAX_VY_DOWN = 15.0


def _norm_vx(v: float) -> float:
    return (v + MAX_VX) / (2 * MAX_VX)


def _norm_vy(v: float) -> float:
    return (v + MAX_VY_UP) / (MAX_VY_UP + MAX_VY_DOWN)


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _player_in_door(p: PlayerState, door: dict) -> bool:
    """True when player hitbox is fully inside door hitbox (mirrors player.js checkDoors)."""
    dhb_x = door["x"] + DOOR_HB_DX
    dhb_y = door["y"] + DOOR_HB_DY
    return (
        p.hb_x >= dhb_x and
        p.hb_x + 36 <= dhb_x + DOOR_HB_W and
        p.hb_y >= dhb_y and
        p.hb_y + 60 <= dhb_y + DOOR_HB_H
    )


class FBWGEnv(gym.Env):
    """Single-level Fireboy-and-Watergirl cooperative environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        level_id: int = 1,
        max_steps: int = 3000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.level_id   = level_id
        self.max_steps  = max_steps
        self.render_mode = render_mode

        self.tiles = load_level_tiles(level_id)
        data = load_level_data(level_id)

        self._fb_start   = data["fireboy_start"]
        self._wg_start   = data["watergirl_start"]
        self._fire_door  = data["fire_door"]
        self._water_door = data["water_door"]

        # Precompute normalised door positions (constant across episode)
        self._fdx_n = self._fire_door["x"]  / CANVAS_W
        self._fdy_n = self._fire_door["y"]  / CANVAS_H
        self._wdx_n = self._water_door["x"] / CANVAS_W
        self._wdy_n = self._water_door["y"] / CANVAS_H

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        self._fb: PlayerState | None = None
        self._wg: PlayerState | None = None
        self._step: int = 0
        self._prev_fb_dist: float = 0.0
        self._prev_wg_dist: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._fb = PlayerState(
            x=float(self._fb_start["x"]),
            y=float(self._fb_start["y"]),
            element="fire",
        )
        self._wg = PlayerState(
            x=float(self._wg_start["x"]),
            y=float(self._wg_start["y"]),
            element="water",
        )
        self._step = 0
        self._prev_fb_dist = self._dist_fb()
        self._prev_wg_dist = self._dist_wg()

        return self._observe(), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        fb_act = int(action[0])
        wg_act = int(action[1])

        self._fb = step_player(self._fb, fb_act, self.tiles)
        self._wg = step_player(self._wg, wg_act, self.tiles)
        self._step += 1

        # Door check
        self._fb.at_door = _player_in_door(self._fb, self._fire_door)
        self._wg.at_door = _player_in_door(self._wg, self._water_door)

        reward, terminated, truncated = self._reward_and_done()

        if self.render_mode == "human":
            self.render()

        return self._observe(), reward, terminated, truncated, {}

    def render(self) -> None:
        fb, wg = self._fb, self._wg
        print(
            f"[{self._step:4d}] "
            f"FB ({fb.x:6.0f},{fb.y:6.0f}) door={fb.at_door} died={fb.died} | "
            f"WG ({wg.x:6.0f},{wg.y:6.0f}) door={wg.at_door} died={wg.died}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        fb, wg = self._fb, self._wg
        return np.array([
            fb.x  / CANVAS_W,
            fb.y  / CANVAS_H,
            _norm_vx(fb.vx),
            _norm_vy(fb.vy),
            wg.x  / CANVAS_W,
            wg.y  / CANVAS_H,
            _norm_vx(wg.vx),
            _norm_vy(wg.vy),
            self._fdx_n,
            self._fdy_n,
            self._wdx_n,
            self._wdy_n,
            float(fb.on_ground),
            float(wg.on_ground),
        ], dtype=np.float32)

    def _dist_fb(self) -> float:
        cx = self._fire_door["x"] + DOOR_HB_DX + DOOR_HB_W / 2
        cy = self._fire_door["y"] + DOOR_HB_DY + DOOR_HB_H / 2
        return _euclidean(self._fb.hb_x + 18, self._fb.hb_y + 30, cx, cy)

    def _dist_wg(self) -> float:
        cx = self._water_door["x"] + DOOR_HB_DX + DOOR_HB_W / 2
        cy = self._water_door["y"] + DOOR_HB_DY + DOOR_HB_H / 2
        return _euclidean(self._wg.hb_x + 18, self._wg.hb_y + 30, cx, cy)

    def _reward_and_done(self) -> tuple[float, bool, bool]:
        # ---- Death ----
        if self._fb.died or self._wg.died:
            return -50.0, True, False

        # ---- Win ----
        if self._fb.at_door and self._wg.at_door:
            return +100.0, True, False

        # ---- Timeout ----
        if self._step >= self.max_steps:
            return -5.0, False, True

        # ---- Dense shaping: progress toward doors ----
        fb_dist = self._dist_fb()
        wg_dist = self._dist_wg()
        shaped = (
            (self._prev_fb_dist - fb_dist) * 0.02 +
            (self._prev_wg_dist - wg_dist) * 0.02
        )
        self._prev_fb_dist = fb_dist
        self._prev_wg_dist = wg_dist

        reward = -0.005 + shaped   # small time penalty + shaping
        return reward, False, False
