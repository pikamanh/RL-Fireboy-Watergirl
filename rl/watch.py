"""
Watch a trained PPO checkpoint control Fireboy and Watergirl.

Example:
    python rl/watch.py --model rl/checkpoints/best_model.zip --level 1
    python rl/watch.py --model rl/checkpoints/best_model.zip --all-levels
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import pygame
from stable_baselines3 import PPO

from env.fbwg_env import FBWGEnv
from env.physics import (
    ACID_PONDS,
    ALL_PONDS,
    BLOCK,
    BS,
    CANVAS_H,
    CANVAS_W,
    FIRE_PONDS,
    GH,
    GW,
    HB_H,
    HB_W,
    SOLID,
    SPR_HB_DX,
    SPR_HB_DY,
    WATER_PONDS,
)
from evaluate import ALL_LEVELS, adapt_obs_for_model, outcome_str


ROOT_DIR = Path(__file__).resolve().parent.parent
MAP_DIR = ROOT_DIR / "res" / "img" / "maps"


def make_window(scale: float) -> tuple[pygame.Surface, pygame.Surface]:
    width = int(CANVAS_W * scale)
    height = int(CANVAS_H * scale)
    screen = pygame.display.set_mode((width, height))
    world = pygame.Surface((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("RL Fireboy & Watergirl Viewer")
    return screen, world


def load_level_map(level_id: int) -> pygame.Surface | None:
    path = MAP_DIR / f"level{level_id}.png"
    if not path.exists():
        return None
    return pygame.image.load(str(path)).convert_alpha()


def draw_tiles(world: pygame.Surface, env: FBWGEnv) -> None:
    for row in range(GH):
        for col in range(GW):
            tile = env.tiles[row * GW + col]
            if tile == 0:
                continue

            rect = pygame.Rect(col * BS, row * BS, BS, BS)
            if tile == BLOCK or tile in SOLID:
                color = (72, 76, 84)
            elif tile in FIRE_PONDS:
                color = (238, 86, 42)
            elif tile in WATER_PONDS:
                color = (46, 139, 219)
            elif tile in ACID_PONDS:
                color = (67, 190, 104)
            elif tile in ALL_PONDS:
                color = (160, 160, 160)
            else:
                color = (92, 96, 104)
            pygame.draw.rect(world, color, rect)


def draw_doors(world: pygame.Surface, env: FBWGEnv) -> None:
    doors = [
        (env._fire_door, (235, 74, 46), env._fb.at_door),
        (env._water_door, (45, 136, 230), env._wg.at_door),
    ]
    for door, color, active in doors:
        rect = pygame.Rect(int(door["x"] + 20), int(door["y"] + 20), 60, 88)
        pygame.draw.rect(world, (24, 24, 28), rect, border_radius=5)
        pygame.draw.rect(world, color, rect.inflate(-8, -8), border_radius=4)
        if active:
            pygame.draw.rect(world, (250, 235, 140), rect, width=5, border_radius=5)


def draw_player(world: pygame.Surface, player, color: tuple[int, int, int]) -> None:
    body = pygame.Rect(int(player.x), int(player.y), 80, 100)
    hitbox = pygame.Rect(int(player.hb_x), int(player.hb_y), HB_W, HB_H)

    shadow = pygame.Rect(body.x + 20, body.bottom - 6, 44, 10)
    pygame.draw.ellipse(world, (0, 0, 0, 90), shadow)
    pygame.draw.rect(world, color, body, border_radius=18)
    pygame.draw.rect(world, (255, 244, 214), body.inflate(-26, -50), border_radius=12)
    pygame.draw.rect(world, (20, 20, 24), hitbox, width=2)

    if player.died:
        pygame.draw.line(world, (20, 20, 24), body.topleft, body.bottomright, 5)
        pygame.draw.line(world, (20, 20, 24), body.topright, body.bottomleft, 5)


def draw_hud(
    world: pygame.Surface,
    font: pygame.font.Font,
    level_id: int,
    episode: int,
    step: int,
    reward: float,
    status: str,
) -> None:
    text = f"Level {level_id} | Episode {episode} | Step {step} | Reward {reward:.1f} | {status}"
    surface = font.render(text, True, (245, 247, 250))
    pad = 12
    bg = pygame.Rect(12, 12, surface.get_width() + pad * 2, surface.get_height() + pad)
    pygame.draw.rect(world, (20, 22, 28), bg, border_radius=6)
    world.blit(surface, (bg.x + pad, bg.y + pad // 2))


def run_episode(
    model: PPO,
    level_id: int,
    screen: pygame.Surface,
    world: pygame.Surface,
    scale: float,
    fps: int,
    max_steps: int,
    episode: int,
    exit_on_done: bool,
    max_render_frames: int | None,
) -> bool:
    env = FBWGEnv(level_id=level_id, max_steps=max_steps)
    level_map = load_level_map(level_id)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    obs, _ = env.reset()
    obs = adapt_obs_for_model(obs, model, level_id)
    done = False
    paused = False
    ep_reward = 0.0
    step = 0
    status = "RUNNING"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    env.close()
                    return False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    env.close()
                    return True
                if event.key == pygame.K_n and done:
                    env.close()
                    return True

        if not paused and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = adapt_obs_for_model(obs, model, level_id)
            ep_reward += reward
            step += 1
            done = terminated or truncated
            if done:
                status = outcome_str(env)
                if exit_on_done:
                    env.close()
                    return False
            if max_render_frames is not None and step >= max_render_frames:
                env.close()
                return False

        world.fill((12, 18, 28))
        if level_map is not None:
            world.blit(level_map, (0, 0))
        else:
            draw_tiles(world, env)
        draw_doors(world, env)
        draw_player(world, env._fb, (231, 70, 43))
        draw_player(world, env._wg, (50, 132, 222))
        draw_hud(world, font, level_id, episode, step, ep_reward, status)

        scaled = pygame.transform.smoothscale(world, screen.get_size())
        screen.blit(scaled, (0, 0))
        pygame.display.flip()
        clock.tick(fps if not done else 15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.72)
    parser.add_argument("--exit-on-done", action="store_true")
    parser.add_argument("--max-render-frames", type=int, default=None)
    args = parser.parse_args()

    levels = ALL_LEVELS if args.all_levels else [args.level]
    model = PPO.load(args.model, device="cpu")

    pygame.init()
    try:
        screen, world = make_window(args.scale)
        episode = 1
        keep_running = True
        while keep_running:
            for level_id in levels:
                for _ in range(args.episodes):
                    keep_running = run_episode(
                        model=model,
                        level_id=level_id,
                        screen=screen,
                        world=world,
                        scale=args.scale,
                        fps=args.fps,
                        max_steps=args.max_steps,
                        episode=episode,
                        exit_on_done=args.exit_on_done,
                        max_render_frames=args.max_render_frames,
                    )
                    episode += 1
                    if not keep_running:
                        break
                if not keep_running:
                    break
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
