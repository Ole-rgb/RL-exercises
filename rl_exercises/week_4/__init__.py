"""Week 4 public exports."""

from __future__ import annotations

from typing import Any

from importlib import import_module

__all__ = [
    "DQNAgent",
    "ExtendedDQNAgent",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "ReplayBuffer",
    "SumTree",
]


def __getattr__(name: str) -> Any:
    """Lazily load week 4 exports to avoid unnecessary heavy imports."""
    if name == "ReplayBuffer":
        return import_module("rl_exercises.week_4.buffers").ReplayBuffer
    if name == "DQNAgent":
        return import_module("rl_exercises.week_4.dqn").DQNAgent
    if name == "QNetwork":
        return import_module("rl_exercises.week_4.networks").QNetwork
    if name == "ExtendedDQNAgent":
        return import_module(
            "rl_exercises.week_4.level_3.extended_dqn"
        ).ExtendedDQNAgent
    if name == "SumTree":
        return import_module("rl_exercises.week_4.level_3.per").SumTree
    if name == "PrioritizedReplayBuffer":
        return import_module("rl_exercises.week_4.level_3.per").PrioritizedReplayBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
