"""
Serve the original browser game and control it with a trained PPO checkpoint.

Run:
    python rl/web_play.py --model rl/checkpoints/level1/best/best_model.zip --level 1

Open the printed URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO

from env.fbwg_env import make_observation_array
from env.level_loader import load_level_data, load_level_tiles
from env.physics import PlayerState
from evaluate import adapt_obs_for_model


ROOT_DIR = Path(__file__).resolve().parent.parent


def _player_from_payload(player: dict, element: str) -> PlayerState:
    return PlayerState(
        x=float(player["x"]),
        y=float(player["y"]),
        element=element,
        vx=float(player["vx"]),
        vy=float(player["vy"]),
        on_ground=bool(player["onGround"]),
    )


def make_observation(payload: dict, model: PPO) -> tuple[np.ndarray, int]:
    level_id = int(payload.get("level", 1))
    players = payload["players"]
    fireboy = _player_from_payload(players[0], "fire")
    watergirl = _player_from_payload(players[1], "water")
    level_data = load_level_data(level_id)

    obs = make_observation_array(
        tiles=load_level_tiles(level_id),
        fb=fireboy,
        wg=watergirl,
        fire_door=level_data["fire_door"],
        water_door=level_data["water_door"],
    )
    return adapt_obs_for_model(obs, model, level_id), level_id


class RLGameHandler(SimpleHTTPRequestHandler):
    model: PPO

    def do_POST(self) -> None:
        if self.path != "/rl/action":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            obs, _ = make_observation(payload, self.model)
            action, _ = self.model.predict(obs, deterministic=True)
            body = json.dumps({"actions": [int(action[0]), int(action[1])]}).encode(
                "utf-8"
            )
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        if self.path == "/rl/action":
            return
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--level", type=int, default=1)
    args = parser.parse_args()

    RLGameHandler.model = PPO.load(args.model, device="cpu")
    handler = partial(RLGameHandler, directory=str(ROOT_DIR))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}/index.html?rl=1&level={args.level}"
    print(f"Serving RL-controlled game at {url}")
    print("Open the URL and the PPO policy will drive both players.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
