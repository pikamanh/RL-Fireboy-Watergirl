"""
Gymnasium environment for Fireboy and Watergirl.

Observation:
    14 base floats:
        fb_x, fb_y, fb_vx, fb_vy,
        wg_x, wg_y, wg_vx, wg_vy,
        fire_door_x, fire_door_y,
        water_door_x, water_door_y,
        fb_on_ground, wg_on_ground
    plus two 7x7 local tile grids, one around each player.

Action:
    MultiDiscrete([6, 6]) = [fireboy_action, watergirl_action]
    0 idle, 1 left, 2 right, 3 jump, 4 jump_left, 5 jump_right
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .level_loader import load_level_data, load_level_tiles
from .physics import (
    ACID_PONDS,
    ACT_JUMP,
    ACT_JUMP_LEFT,
    ACT_JUMP_RIGHT,
    BS,
    CANVAS_H,
    CANVAS_W,
    EMPTY,
    FIRE_PONDS,
    GH,
    GW,
    HB_H,
    HB_W,
    JUMP_VY,
    MOVE_VX,
    N_ACTIONS,
    PlayerState,
    SOLID,
    WATER_PONDS,
    step_player,
)

# Door geometry mirrors res/js/ingameAssets/door.js.
DOOR_HB_DX = 20
DOOR_HB_DY = 20
DOOR_HB_W = 60
DOOR_HB_H = 88

MAX_VX = MOVE_VX + 0.5
MAX_VY_UP = abs(JUMP_VY) + 1.0
MAX_VY_DOWN = 15.0

LOCAL_RADIUS = 5
LOCAL_GRID = LOCAL_RADIUS * 2 + 1
BASE_OBS_DIM = 14
OBS_DIM = BASE_OBS_DIM + 2 * LOCAL_GRID * LOCAL_GRID

JUMP_ACTIONS = frozenset({ACT_JUMP, ACT_JUMP_LEFT, ACT_JUMP_RIGHT})


def _norm_vx(v: float) -> float:
    return (v + MAX_VX) / (2 * MAX_VX)


def _norm_vy(v: float) -> float:
    return (v + MAX_VY_UP) / (MAX_VY_UP + MAX_VY_DOWN)


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _player_in_door(p: PlayerState, door: dict) -> bool:
    dhb_x = door["x"] + DOOR_HB_DX
    dhb_y = door["y"] + DOOR_HB_DY
    return (
        p.hb_x >= dhb_x
        and p.hb_x + HB_W <= dhb_x + DOOR_HB_W
        and p.hb_y >= dhb_y
        and p.hb_y + HB_H <= dhb_y + DOOR_HB_H
    )


def _tile_value(tile: int, element: str) -> float:
    if tile == EMPTY:
        return 0.0
    if tile in SOLID:
        return 0.25
    if tile in ACID_PONDS:
        return 1.0
    if tile in FIRE_PONDS:
        return 0.45 if element == "fire" else 0.9
    if tile in WATER_PONDS:
        return 0.45 if element == "water" else 0.9
    return 0.15


def local_tile_observation(tiles: list[int], player: PlayerState) -> np.ndarray:
    center_col = int((player.hb_x + HB_W / 2) // BS)
    center_row = int((player.hb_y + HB_H / 2) // BS)
    values: list[float] = []

    for row_offset in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
        for col_offset in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1):
            row = center_row + row_offset
            col = center_col + col_offset
            if row < 0 or row >= GH or col < 0 or col >= GW:
                values.append(1.0)
            else:
                values.append(_tile_value(tiles[row * GW + col], player.element))

    return np.array(values, dtype=np.float32)


def make_observation_array(
    tiles: list[int],
    fb: PlayerState,
    wg: PlayerState,
    fire_door: dict,
    water_door: dict,
) -> np.ndarray:
    base = np.array(
        [
            fb.x / CANVAS_W,
            fb.y / CANVAS_H,
            _norm_vx(fb.vx),
            _norm_vy(fb.vy),
            wg.x / CANVAS_W,
            wg.y / CANVAS_H,
            _norm_vx(wg.vx),
            _norm_vy(wg.vy),
            fire_door["x"] / CANVAS_W,
            fire_door["y"] / CANVAS_H,
            water_door["x"] / CANVAS_W,
            water_door["y"] / CANVAS_H,
            float(fb.on_ground),
            float(wg.on_ground),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base,
            local_tile_observation(tiles, fb),
            local_tile_observation(tiles, wg),
        ]
    ).astype(np.float32)


class FBWGEnv(gym.Env):
    """Single-level cooperative Fireboy-and-Watergirl environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        level_id: int = 1,
        max_steps: int = 3000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.level_id = level_id
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.tiles = load_level_tiles(level_id)
        data = load_level_data(level_id)
        self._fb_start = data["fireboy_start"]
        self._wg_start = data["watergirl_start"]
        self._fire_door = data["fire_door"]
        self._water_door = data["water_door"]

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS, N_ACTIONS])

        self._fb: PlayerState | None = None
        self._wg: PlayerState | None = None
        self._step = 0
        self._prev_fb_dist = 0.0
        self._prev_wg_dist = 0.0
        self._prev_fb_x = 0.0
        self._prev_wg_x = 0.0
        self._stalled_steps = 0
        self._fb_door_bonus_given = False
        self._wg_door_bonus_given = False
        self._last_action = (0, 0)

        self._best_fb_dist = float('inf')
        self._best_wg_dist = float('inf')
        self._no_progress_steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
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
        self._prev_fb_x = self._fb.x
        self._prev_wg_x = self._wg.x
        self._stalled_steps = 0
        self._fb_door_bonus_given = False
        self._wg_door_bonus_given = False
        self._last_action = (0, 0)

        # Thêm 3 dòng này để reset logic phạt câu giờ cho ván mới:
        self._best_fb_dist = float('inf')
        self._best_wg_dist = float('inf')
        self._no_progress_steps = 0

        return self._observe(), {}

    def step(self, action):
        fb_act = int(action[0])
        wg_act = int(action[1])
        self._last_action = (fb_act, wg_act)

        self._fb = step_player(self._fb, fb_act, self.tiles)
        self._wg = step_player(self._wg, wg_act, self.tiles)
        self._step += 1

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

    def _observe(self) -> np.ndarray:
        return make_observation_array(
            self.tiles, self._fb, self._wg, self._fire_door, self._water_door
        )

    def _dist_fb(self) -> float:
        cx = self._fire_door["x"] + DOOR_HB_DX + DOOR_HB_W / 2
        cy = self._fire_door["y"] + DOOR_HB_DY + DOOR_HB_H / 2
        return _euclidean(self._fb.hb_x + HB_W / 2, self._fb.hb_y + HB_H / 2, cx, cy)

    def _dist_wg(self) -> float:
        cx = self._water_door["x"] + DOOR_HB_DX + DOOR_HB_W / 2
        cy = self._water_door["y"] + DOOR_HB_DY + DOOR_HB_H / 2
        return _euclidean(self._wg.hb_x + HB_W / 2, self._wg.hb_y + HB_H / 2, cx, cy)

    def _reward_and_done(self) -> tuple[float, bool, bool]:
        if self._fb.died or self._wg.died:
            return -20.0, True, False # Giảm phạt chết để khuyến khích thử nghiệm

        if self._fb.at_door and self._wg.at_door:
            return 150.0, True, False

        if self._step >= self.max_steps:
            return -50.0, False, True # Tăng phạt hết giờ để ép phải kết thúc game

        fb_dist = self._dist_fb()
        wg_dist = self._dist_wg()

        # 1. TÍNH PHẦN THƯỞNG CƠ BẢN TRƯỚC (Khởi tạo biến reward)
        shaped = (self._prev_fb_dist - fb_dist) * 0.05 + (
            self._prev_wg_dist - wg_dist
        ) * 0.05
        reward = -0.01 + shaped

        # 2. KIỂM TRA TIẾN ĐỘ (LOGIC CHỐNG CÂU GIỜ)
        if fb_dist < self._best_fb_dist or wg_dist < self._best_wg_dist:
            self._best_fb_dist = min(self._best_fb_dist, fb_dist)
            self._best_wg_dist = min(self._best_wg_dist, wg_dist)
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        if self._no_progress_steps > 150:
            reward -= 0.1  # Bây giờ reward đã tồn tại nên không bị lỗi nữa
        if self._no_progress_steps > 300:
            return -30.0, True, False # Chết luôn vì tội câu giờ, ép ván mới

        # 3. THƯỞNG ĐẾN CỬA
        if self._fb.at_door and not self._fb_door_bonus_given:
            reward += 25.0
            self._fb_door_bonus_given = True
        if self._wg.at_door and not self._wg_door_bonus_given:
            reward += 25.0
            self._wg_door_bonus_given = True

        # 4. PHẠT NHẢY VÔ ÍCH
        if self._last_action[0] == ACT_JUMP:
            reward -= 0.02
        if self._last_action[1] == ACT_JUMP:
            reward -= 0.02

        # Cập nhật lại vị trí cũ cho step tiếp theo
        self._prev_fb_dist = fb_dist
        self._prev_wg_dist = wg_dist
        self._prev_fb_x = self._fb.x
        self._prev_wg_x = self._wg.x

        return reward, False, False