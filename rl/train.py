"""
Train a PPO agent on Fireboy and Watergirl.

Single-level (original behaviour):
    python train.py                        # level 1, 2M steps
    python train.py --level 2 --steps 5000000
    LEVEL=3 python train.py

All-levels (multi-task):
    python train.py --all-levels                         # uniform sampling
    python train.py --all-levels --sampling plr          # Prioritized Level Replay
    python train.py --all-levels --sampling curriculum   # progressive unlock
    python train.py --all-levels --steps 10000000        # recommended for all levels
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.fbwg_env import FBWGEnv
from env.multi_level_env import MultiLevelEnv, PLRState
from env.plr_callback import PLRCallback

ALL_LEVELS = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Env factory helpers
# ---------------------------------------------------------------------------

def make_single_env(level_id: int, max_steps: int = 3000):
    """Factory for a single-level environment (original behaviour)."""
    def _init():
        return Monitor(FBWGEnv(level_id=level_id, max_steps=max_steps))
    return _init


def make_multi_env(
    level_ids: list[int],
    max_steps: int = 3000,
    plr_state: PLRState | None = None,
):
    """Factory for a multi-level environment."""
    def _init():
        return Monitor(
            MultiLevelEnv(
                level_ids=level_ids,
                max_steps=max_steps,
                plr_state=plr_state,
            )
        )
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    # Mode
    parser.add_argument(
        "--all-levels", action="store_true",
        help="Train on all 6 levels (multi-task). Overrides --level.",
    )
    parser.add_argument(
        "--level", type=int, default=int(os.environ.get("LEVEL", 1)),
        help="Level ID when training a single level (ignored with --all-levels).",
    )
    parser.add_argument(
        "--sampling", choices=["uniform", "plr", "curriculum"], default="plr",
        help="Level sampling strategy for --all-levels (default: plr).",
    )

    # Training
    parser.add_argument("--steps",  type=int, default=None,
                        help="Total env steps (default 2M single / 10M all-levels).")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--load",   type=str, default=None,
                        help="Path to existing model ZIP to continue training.")

    # PLR / Curriculum knobs
    parser.add_argument("--curriculum-threshold", type=float, default=-10.0,
                        help="Mean return needed to unlock next level in curriculum mode.")
    parser.add_argument("--plr-window", type=int, default=20,
                        help="Trailing episode window for PLR score estimation.")

    args = parser.parse_args()

    # ---- Resolve defaults -----------------------------------------------
    multi = args.all_levels
    default_steps = 10_000_000 if multi else 2_000_000
    total_steps = args.steps if args.steps is not None else default_steps

    # ---- Build environments -----------------------------------------------
    if multi:
        plr_state = PLRState(
            level_ids=ALL_LEVELS,
            mode=args.sampling,
            window=args.plr_window,
            curriculum_threshold=args.curriculum_threshold,
            verbose=True,
        )
        train_env = make_vec_env(
            make_multi_env(ALL_LEVELS, plr_state=plr_state),
            n_envs=args.n_envs,
        )
        eval_env = make_vec_env(
            make_multi_env(ALL_LEVELS, max_steps=3000),   # uniform for eval
            n_envs=1,
        )
        ckpt_dir = "./checkpoints/all_levels"
        log_dir  = "./logs/all_levels/"
        save_name = f"ppo_fbwg_all_levels_final"
        print(
            f"[Multi-level] sampling={args.sampling}  "
            f"levels={ALL_LEVELS}  steps={total_steps:,}  n_envs={args.n_envs}"
        )
    else:
        plr_state = None
        level = args.level
        train_env = make_vec_env(make_single_env(level), n_envs=args.n_envs)
        eval_env  = make_vec_env(make_single_env(level, max_steps=3000), n_envs=1)
        ckpt_dir  = f"./checkpoints/level{level}"
        log_dir   = f"./logs/level{level}/"
        save_name = f"ppo_fbwg_level{level}_final"
        print(
            f"[Single-level] level={level}  steps={total_steps:,}  n_envs={args.n_envs}"
        )

    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Callbacks -------------------------------------------------------
    callbacks = []

    callbacks.append(CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=ckpt_dir,
        name_prefix="ppo_fbwg",
        verbose=1,
    ))
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=f"{ckpt_dir}/best/",
        log_path=f"{log_dir}eval/",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=10 if multi else 5,
        deterministic=True,
        verbose=1,
    ))
    if plr_state is not None:
        callbacks.append(PLRCallback(plr_state, log_freq=20_000, verbose=1))

    # ---- Model -----------------------------------------------------------
    # Larger network for multi-task to accommodate 6 level contexts
    net_arch = [512, 256] if multi else [256, 256]

    if args.load:
        print(f"Loading model from {args.load}")
        model = PPO.load(args.load, env=train_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128 if multi else 64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02 if multi else 0.01,   # more exploration for multi-task
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=log_dir,
            verbose=1,
        )

    # ---- Train -----------------------------------------------------------
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    model.save(save_name)
    print(f"Saved: {save_name}.zip")


if __name__ == "__main__":
    main()
