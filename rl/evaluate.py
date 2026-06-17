"""
Evaluate a trained Fireboy-Watergirl PPO checkpoint.

Single level:
    python rl/evaluate.py --model ppo_fbwg_level1_final.zip --level 1 --episodes 5

All levels:
    python rl/evaluate.py --model ppo_fbwg_all_levels_final.zip --all-levels
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))

from env.fbwg_env import FBWGEnv


ALL_LEVELS = [1, 2, 3, 4, 5, 6]


def outcome_str(env: FBWGEnv) -> str:
    if env._fb.at_door and env._wg.at_door:
        return "WIN"
    if env._fb.died or env._wg.died:
        return "DEATH"
    return "TIMEOUT"


def model_obs_dim(model: PPO) -> int:
    return int(model.observation_space.shape[0])


def level_feature(level_id: int) -> float:
    span = max(max(ALL_LEVELS) - min(ALL_LEVELS), 1)
    return (level_id - min(ALL_LEVELS)) / span


def adapt_obs_for_model(obs: np.ndarray, model: PPO, level_id: int) -> np.ndarray:
    expected_dim = model_obs_dim(model)
    if obs.shape == (expected_dim,):
        return obs
    if obs.shape == (expected_dim - 1,):
        return np.concatenate(
            [obs, np.array([level_feature(level_id)], dtype=np.float32)]
        )
    raise ValueError(
        f"Model expects observation shape ({expected_dim},), "
        f"but env returned {obs.shape}. Train a new checkpoint after env changes."
    )


def run_single_level(
    model: PPO,
    level_id: int,
    episodes: int,
    max_steps: int,
    render: bool,
    log_every: int,
) -> dict:
    env = FBWGEnv(level_id=level_id, max_steps=max_steps, render_mode=None)
    wins, total_rewards, steps_list = 0, [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        obs = adapt_obs_for_model(obs, model, level_id)
        ep_reward, step, done = 0.0, 0, False
        print(f"\n=== Level {level_id} | Episode {ep} ===")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = adapt_obs_for_model(obs, model, level_id)
            ep_reward += reward
            step += 1
            done = terminated or truncated

            if render and (step % log_every == 0 or done):
                env.render()

        oc = outcome_str(env)
        if oc == "WIN":
            wins += 1
        total_rewards.append(ep_reward)
        steps_list.append(step)
        print(f"  {oc}  steps={step}  reward={ep_reward:.2f}")

    env.close()
    return {
        "level": level_id,
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_steps": float(np.mean(steps_list)),
    }


def run_all_levels(
    model: PPO,
    episodes_per_level: int,
    max_steps: int,
    render: bool,
    log_every: int,
) -> None:
    results = []
    for level_id in ALL_LEVELS:
        stats = run_single_level(
            model, level_id, episodes_per_level, max_steps, render, log_every
        )
        results.append(stats)

    print("\n" + "=" * 60)
    print(f"{'Level':<8}{'Win Rate':>10}{'Mean Reward':>14}{'Mean Steps':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"  {r['level']:<6}{r['win_rate']:>9.1%}"
            f"{r['mean_reward']:>14.2f}{r['mean_steps']:>12.1f}"
        )
    print("=" * 60)
    overall_wr = np.mean([r["win_rate"] for r in results])
    print(f"  {'Overall':<6}{overall_wr:>9.1%}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episodes-per-level", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    log_every = max(args.log_every, 1)

    if args.all_levels:
        run_all_levels(
            model,
            args.episodes_per_level,
            args.max_steps,
            args.render,
            log_every,
        )
    else:
        stats = run_single_level(
            model,
            args.level,
            args.episodes,
            args.max_steps,
            args.render,
            log_every,
        )
        print(
            f"\nWin rate: {stats['win_rate']:.1%} "
            f"Mean reward: {stats['mean_reward']:.2f}"
        )


if __name__ == "__main__":
    main()
