"""
Physics engine – faithful Python port of the JS player physics.

Sources used:
  res/js/player.js  → gravity(), horizontalCollision(), verticalCollision()
  res/js/game.js    → velocity.x = ±2, jump vy = -4.35
  res/js/helpers.js → GAME_SIZE (39×29 blocks, 36 px each)
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Grid / canvas constants
# ---------------------------------------------------------------------------
BS = 36          # block size in pixels
GW = 39          # grid width  (columns)
GH = 29          # grid height (rows)

CANVAS_W = GW * BS   # 1404
CANVAS_H = GH * BS   # 1044

# ---------------------------------------------------------------------------
# Tile type IDs  (mirrors collisionBlocks.js)
# ---------------------------------------------------------------------------
EMPTY               = 0
BLOCK               = 1
TRIANGLE_LU         = 2   # left-up
TRIANGLE_RU         = 3   # right-up
TRIANGLE_LD         = 4   # left-down
TRIANGLE_RD         = 5   # right-down
FIRE_POND           = 6
FIRE_POND_TL        = 7
FIRE_POND_TR        = 8
WATER_POND          = 9
WATER_POND_TL       = 10
WATER_POND_TR       = 11
ACID_POND           = 12
ACID_POND_TL        = 13
ACID_POND_TR        = 14

# Solid = blocks that stop player movement (triangles treated as solid in v1)
SOLID = frozenset({BLOCK, TRIANGLE_LU, TRIANGLE_RU, TRIANGLE_LD, TRIANGLE_RD})
FIRE_PONDS  = frozenset({FIRE_POND,  FIRE_POND_TL,  FIRE_POND_TR})
WATER_PONDS = frozenset({WATER_POND, WATER_POND_TL, WATER_POND_TR})
ACID_PONDS  = frozenset({ACID_POND,  ACID_POND_TL,  ACID_POND_TR})
ALL_PONDS   = FIRE_PONDS | WATER_PONDS | ACID_PONDS

# ---------------------------------------------------------------------------
# Player geometry  (mirrors player.js hitboxPositionCalc())
# ---------------------------------------------------------------------------
SPR_HB_DX = 31    # sprite → hitbox offset
SPR_HB_DY = 37
HB_W      = 36    # hitbox size
HB_H      = 60
LEG_DX    = 12    # hitbox → legs offset  ( = (36-12)/2 )
LEG_DY    = 36    #                       ( = 60-24 )
LEG_W     = 12
LEG_H     = 24

# ---------------------------------------------------------------------------
# Movement constants  (from game.js keydown handler)
# ---------------------------------------------------------------------------
MOVE_VX  =  2.0
JUMP_VY  = -4.35

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ACT_IDLE  = 0
ACT_LEFT  = 1
ACT_RIGHT = 2
ACT_JUMP  = 3
N_ACTIONS = 4


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------
@dataclass
class PlayerState:
    x:         float
    y:         float
    element:   str           # "fire" | "water"
    vx:        float = 0.0
    vy:        float = 0.0
    on_ground: bool  = False
    died:      bool  = False
    at_door:   bool  = False

    # Derived hitbox helpers (read-only shortcuts)
    @property
    def hb_x(self) -> float:  return self.x + SPR_HB_DX
    @property
    def hb_y(self) -> float:  return self.y + SPR_HB_DY
    @property
    def leg_x(self) -> float: return self.hb_x + LEG_DX
    @property
    def leg_y(self) -> float: return self.hb_y + LEG_DY

    def copy(self) -> "PlayerState":
        return PlayerState(
            x=self.x, y=self.y, element=self.element,
            vx=self.vx, vy=self.vy,
            on_ground=self.on_ground, died=self.died, at_door=self.at_door,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_solid_blocks(tiles: list[int], rx: float, ry: float, rw: float, rh: float):
    """Yield (bx, by, tile) for every non-empty tile overlapping the rectangle."""
    col0 = max(0, int(rx // BS))
    col1 = min(GW - 1, int((rx + rw - 1) // BS))
    row0 = max(0, int(ry // BS))
    row1 = min(GH - 1, int((ry + rh - 1) // BS))
    for r in range(row0, row1 + 1):
        for c in range(col0, col1 + 1):
            t = tiles[r * GW + c]
            if t != EMPTY:
                yield c * BS, r * BS, t


# ---------------------------------------------------------------------------
# Physics step
# ---------------------------------------------------------------------------
def step_player(s: PlayerState, action: int, tiles: list[int]) -> PlayerState:
    """
    Advance one frame (≈1/60 s).
    Returns a NEW PlayerState; input is not mutated.
    """
    if s.died:
        return s.copy()

    p = s.copy()

    # ------------------------------------------------------------------ #
    # 1. Apply horizontal action (sets vx each frame, not accumulated)
    # ------------------------------------------------------------------ #
    if action == ACT_LEFT:
        p.vx = -MOVE_VX
    elif action == ACT_RIGHT:
        p.vx = MOVE_VX
    else:
        p.vx = 0.0

    if action == ACT_JUMP and p.on_ground:
        p.vy = JUMP_VY

    # ------------------------------------------------------------------ #
    # 2. Move horizontally, then resolve solid-block collisions
    # ------------------------------------------------------------------ #
    p.x += p.vx
    _resolve_horizontal(p, tiles)

    # ------------------------------------------------------------------ #
    # 3. Apply gravity (port of player.js gravity())
    # ------------------------------------------------------------------ #
    if p.vy < 0:
        p.vy += 0.07
        if p.vy > -0.001:
            p.vy = 0.0
    elif p.vy > 0:
        p.vy += 0.02 if p.vy >= 1.6 else 0.07
    else:
        p.vy = 2.02   # start of free-fall

    p.y += p.vy

    # ------------------------------------------------------------------ #
    # 4. Resolve vertical solid-block collisions
    # ------------------------------------------------------------------ #
    _resolve_vertical(p, tiles)

    # ------------------------------------------------------------------ #
    # 5. Pond / acid death check
    # ------------------------------------------------------------------ #
    _check_ponds(p, tiles)

    return p


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------
def _resolve_horizontal(p: PlayerState, tiles: list[int]) -> None:
    if p.vx == 0:
        return

    hb_x = p.hb_x
    hb_y = p.hb_y

    for bx, by, tile in _iter_solid_blocks(tiles, hb_x, hb_y, HB_W, HB_H):
        if tile not in SOLID:
            continue
        # Vertical overlap (must share y range)
        if not (hb_y + HB_H > by + 1 and hb_y < by + BS):
            continue
        # Horizontal overlap
        if not (hb_x < bx + BS and hb_x + HB_W > bx):
            continue

        if p.vx > 0:   # moving right → push left
            p.x = bx - (SPR_HB_DX + HB_W) - 0.01
        else:           # moving left  → push right
            p.x = bx + BS - SPR_HB_DX + 0.01

        p.vx = 0.0
        break  # one resolution per frame is enough


def _resolve_vertical(p: PlayerState, tiles: list[int]) -> None:
    p.on_ground = False
    hb_x = p.hb_x
    hb_y = p.hb_y

    for bx, by, tile in _iter_solid_blocks(tiles, hb_x, hb_y, HB_W, HB_H):
        if tile not in SOLID:
            continue
        # Horizontal overlap (hitbox must share x range)
        if not (hb_x < bx + BS and hb_x + HB_W > bx):
            continue

        if p.vy >= 0:
            # Falling / standing: legs feet hit top of block
            leg_x = p.leg_x
            if (leg_x < bx + BS and leg_x + LEG_W > bx and
                    hb_y + HB_H >= by and hb_y + HB_H <= by + BS):
                p.y = by - (SPR_HB_DY + HB_H) - 0.01
                p.vy = 0.0
                p.on_ground = True
                break

        else:
            # Rising: head hits ceiling
            if hb_y <= by + BS and hb_y >= by:
                p.y = by + BS - SPR_HB_DY + 0.01
                p.vy = 0.0
                break


def _check_ponds(p: PlayerState, tiles: list[int]) -> None:
    """Kill player if they touch a harmful pond element."""
    hb_x = p.hb_x
    hb_y = p.hb_y

    for bx, by, tile in _iter_solid_blocks(tiles, hb_x, hb_y, HB_W, HB_H):
        if tile not in ALL_PONDS:
            continue
        # Death zone: lower portion of pond block
        if not (hb_y + HB_H >= by + 10 and hb_y + HB_H <= by + BS):
            continue

        if tile in FIRE_PONDS  and p.element != "fire":
            p.died = True
        elif tile in WATER_PONDS and p.element != "water":
            p.died = True
        elif tile in ACID_PONDS:
            p.died = True

        if p.died:
            break
