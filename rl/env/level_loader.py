"""Parse level tile arrays from collisionBlocks.js and load entity data from JSON."""

import re
import json
import pathlib

GAME_DIR = pathlib.Path(__file__).parent.parent.parent
JS_PATH = GAME_DIR / "res" / "js" / "collisionBlocks.js"
DATA_DIR = GAME_DIR / "res" / "data"

# Cached so we only parse the JS file once per process
_tiles_cache: dict[int, list[int]] = {}
_constants_cache: dict[str, int] | None = None


def _get_constants() -> dict[str, int]:
    global _constants_cache
    if _constants_cache is not None:
        return _constants_cache
    text = JS_PATH.read_text()
    _constants_cache = {
        m.group(1): int(m.group(2))
        for m in re.finditer(r"const\s+([A-Z_]+)\s*=\s*(\d+)", text)
    }
    return _constants_cache


def load_level_tiles(level_id: int) -> list[int]:
    """Return flat tile array (row-major, 39×29) for the given level."""
    if level_id in _tiles_cache:
        return _tiles_cache[level_id]

    text = JS_PATH.read_text()
    constants = _get_constants()

    # Match the level array between [ and the matching ]
    m = re.search(
        rf"const\s+level{level_id}\s*=\s*\[(.*?)\]",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"Level {level_id} not found in collisionBlocks.js")

    body = m.group(1)
    # Strip JS line comments
    body = re.sub(r"//[^\n]*", "", body)
    # Replace constant names with their integer values (longest-first to avoid partial matches)
    for name in sorted(constants, key=len, reverse=True):
        body = re.sub(rf"\b{name}\b", str(constants[name]), body)

    tiles = [int(x) for x in re.split(r"[\s,]+", body.strip()) if x]
    _tiles_cache[level_id] = tiles
    return tiles


def load_level_data(level_id: int) -> dict:
    """Return start positions for both players and door positions for a level."""
    lvl = str(level_id)

    players = json.loads((DATA_DIR / "players.json").read_text())
    doors = json.loads((DATA_DIR / "doors.json").read_text())

    fire_door = next(d["position"] for d in doors[lvl] if d["element"] == "fire")
    water_door = next(d["position"] for d in doors[lvl] if d["element"] == "water")

    return {
        "fireboy_start": players["fireboy"][lvl]["position"],
        "watergirl_start": players["watergirl"][lvl]["position"],
        "fire_door": fire_door,
        "water_door": water_door,
    }
