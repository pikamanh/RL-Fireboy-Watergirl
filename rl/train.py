"""
Train PPO agents for Fireboy and Watergirl.

Recommended workflow:
    python rl/train.py --level 1 --steps 2000000
    python rl/evaluate.py --model rl/checkpoints/level1/best/best_model.zip --level 1

After single levels work:
    python rl/train.py --all-levels --sampling curriculum --steps 10000000
"""

from __future__ import annotations

import argparse
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(__file__))

from env.fbwg_env import FBWGEnv
from env.multi_level_env import MultiLevelEnv, PLRState
from env.plr_callback import PLRCallback


ALL_LEVELS = [1, 2, 3, 4, 5, 6]


def make_single_env(level_id: int, max_steps: int):
    def _init():
        return Monitor(FBWGEnv(level_id=level_id, max_steps=max_steps))

    return _init


def make_multi_env(level_ids: list[int], max_steps: int, plr_state: PLRState | None = None):
    def _init():
        return Monitor(
            MultiLevelEnv(level_ids=level_ids, max_steps=max_steps, plr_state=plr_state)
        )

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--level", type=int, default=int(os.environ.get("LEVEL", 1)))
    parser.add_argument(
        "--sampling",
        choices=["uniform", "plr", "curriculum"],
        default="curriculum",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--curriculum-threshold", type=float, default=25.0)
    parser.add_argument("--plr-window", type=int, default=20)
    args = parser.parse_args()

    multi = args.all_levels
    total_steps = args.steps or (10_000_000 if multi else 2_000_000)

    if multi:
        plr_state = PLRState(
            level_ids=ALL_LEVELS,
            mode=args.sampling,
            window=args.plr_window,
            curriculum_threshold=args.curriculum_threshold,
            verbose=True,
        )
        train_env = make_vec_env(
            make_multi_env(ALL_LEVELS, args.max_steps, plr_state=plr_state),
            n_envs=args.n_envs,
        )
        eval_env = make_vec_env(
            make_multi_env(ALL_LEVELS, args.max_steps),
            n_envs=1,
        )
        ckpt_dir = "rl/checkpoints/all_levels"
        log_dir = "rl/logs/all_levels"
        save_name = "rl/checkpoints/ppo_fbwg_all_levels_final"
        print(
            f"[Multi-level] sampling={args.sampling} levels={ALL_LEVELS} "
            f"steps={total_steps:,} n_envs={args.n_envs}"
        )
    else:
        plr_state = None
        train_env = make_vec_env(
            make_single_env(args.level, args.max_steps),
            n_envs=args.n_envs,
        )
        eval_env = make_vec_env(
            make_single_env(args.level, args.max_steps),
            n_envs=1,
        )
        ckpt_dir = f"rl/checkpoints/level{args.level}"
        log_dir = f"rl/logs/level{args.level}"
        save_name = f"rl/checkpoints/ppo_fbwg_level{args.level}_final"
        print(
            f"[Single-level] level={args.level} steps={total_steps:,} "
            f"n_envs={args.n_envs}"
        )

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=ckpt_dir,
            name_prefix="ppo_fbwg",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{ckpt_dir}/best",
            log_path=f"{log_dir}/eval",
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=10 if multi else 5,
            deterministic=True,
            verbose=1,
        ),
    ]
    if plr_state is not None:
        callbacks.append(PLRCallback(plr_state, log_freq=20_000, verbose=1))

    net_arch = [512, 256] if multi else [256, 256]
    if args.load:
        print(f"Loading model {args.load}")
        model = PPO.load(args.load, env=train_env, device="cpu")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256 if multi else 128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03 if multi else 0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=log_dir,
            device="cpu",
            verbose=1,
        )

    model.learn(total_timesteps=total_steps, callback=callbacks, progress_bar=True)
    model.save(save_name)
    print(f"Saved: {save_name}.zip")
    print(f"Best eval checkpoint: {ckpt_dir}/best/best_model.zip")


if __name__ == "__main__":
    main()
