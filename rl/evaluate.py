"""
Run a trained agent and report per-episode results.

Single-level:
    python evaluate.py --model ppo_fbwg_level1_final --level 1 --episodes 5

All-levels model:
    python evaluate.py --model ppo_fbwg_all_levels_final --all-levels
    python evaluate.py --model ppo_fbwg_all_levels_final --all-levels --episodes-per-level 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from stable_baselines3 import PPO

from env.fbwg_env import FBWGEnv
from env.multi_level_env import MultiLevelEnv

ALL_LEVELS = [1, 2, 3, 4, 5, 6]


def outcome_str(env: FBWGEnv) -> str:
    if env._fb.at_door and env._wg.at_door:
        return "WIN"
    if env._fb.died or env._wg.died:
        return "DEATH"
    return "TIMEOUT"


def run_single_level(model: PPO, level_id: int, episodes: int, max_steps: int) -> dict:
    env = FBWGEnv(level_id=level_id, max_steps=max_steps, render_mode="human")
    wins, total_rewards, steps_list = 0, [], []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward, step, done = 0.0, 0, False
        print(f"\n=== Level {level_id} | Episode {ep} ===")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

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


def run_all_levels(model: PPO, episodes_per_level: int, max_steps: int) -> None:
    results = []
    for level_id in ALL_LEVELS:
        stats = run_single_level(model, level_id, episodes_per_level, max_steps)
        results.append(stats)

    # Summary table
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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--model",              required=True)
    parser.add_argument("--level",              type=int, default=1)
    parser.add_argument("--all-levels",         action="store_true",
                        help="Evaluate the model on all 6 levels.")
    parser.add_argument("--episodes",           type=int, default=5,
                        help="Episodes (single-level mode).")
    parser.add_argument("--episodes-per-level", type=int, default=5,
                        help="Episodes per level (--all-levels mode).")
    parser.add_argument("--max-steps",          type=int, default=3000)
    args = parser.parse_args()

    model = PPO.load(args.model)

    if args.all_levels:
        run_all_levels(model, args.episodes_per_level, args.max_steps)
    else:
        stats = run_single_level(model, args.level, args.episodes, args.max_steps)
        print(f"\nWin rate: {stats['win_rate']:.1%}  Mean reward: {stats['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
