"""Callback that logs per-level PLR statistics to TensorBoard."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .multi_level_env import PLRState


class PLRCallback(BaseCallback):
    """
    Reads a shared PLRState and logs per-level stats to TensorBoard
    every `log_freq` timesteps.

    Metrics logged:
        plr/levelN_mean_return  – trailing mean episode return for level N
        plr/levelN_episodes     – total episodes completed on level N
        plr/n_unlocked          – number of levels currently available (curriculum mode)
    """

    def __init__(self, plr_state: PLRState, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.plr_state = plr_state
        self.log_freq = log_freq
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.log_freq:
            return True
        self._last_log = self.num_timesteps

        summary = self.plr_state.summary()
        for level_id, stats in summary.items():
            if stats["mean_return"] is not None:
                self.logger.record(
                    f"plr/level{level_id}_mean_return",
                    stats["mean_return"],
                )
            self.logger.record(f"plr/level{level_id}_episodes", stats["episodes"])

        n_unlocked = sum(1 for s in summary.values() if s["available"])
        self.logger.record("plr/n_unlocked", n_unlocked)

        if self.verbose >= 1:
            print(f"\n[PLR @ {self.num_timesteps:,}]")
            for level_id, stats in summary.items():
                ret_str = (
                    f"{stats['mean_return']:+.2f}"
                    if stats["mean_return"] is not None
                    else "  N/A "
                )
                avail = "✓" if stats["available"] else "✗"
                print(
                    f"  Level {level_id} [{avail}] "
                    f"return={ret_str}  episodes={stats['episodes']:4d}"
                )

        return True
