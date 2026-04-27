from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from rl_exercises.environments import MarsRover
from rl_exercises.week_2.value_iteration import ValueIteration


class ContextualValueIteration(ValueIteration):
    """
    Value iteration that replans when the environment context changes.

    The agent expects the environment to expose context either through the
    ``info`` dict under ``"context"`` or via a ``get_context()`` method.
    """

    def __init__(
        self,
        env: MarsRover | gymnasium.Env,
        gamma: float = 0.9,
        seed: int = 333,
        **kwargs: dict,
    ) -> None:
        super().__init__(env=env, gamma=gamma, seed=seed, **kwargs)
        self._last_context_signature: tuple[tuple[str, float], ...] | None = None

    def _normalize_context(
        self, context: dict[str, Any] | None
    ) -> tuple[tuple[str, float], ...] | None:
        if not context:
            return None
        return tuple(sorted((str(k), float(v)) for k, v in context.items()))

    def _get_env_context(self) -> dict[str, Any] | None:
        if hasattr(self.env, "get_context"):
            return self.env.get_context()  # type: ignore[no-any-return]
        return None

    def _refresh_mdp(self) -> None:
        self.S = self.env.states
        self.A = self.env.actions
        self.T = self.env.transition_matrix
        self.R_sa = self.env.get_reward_per_action()
        self.n_states, self.n_actions = self.R_sa.shape
        self.V = np.zeros(self.n_states, dtype=float)
        self.pi = np.zeros(self.n_states, dtype=int)
        self.policy_fitted = False

    def _apply_context_to_env(self, context: dict[str, Any] | None) -> None:
        if context is None or not hasattr(self.env, "set_context"):
            return

        if "tilt_angle" in context and "friction" in context:
            self.env.set_context(  # type: ignore[attr-defined]
                tilt_angle=float(context["tilt_angle"]),
                friction=float(context["friction"]),
            )

    def _ensure_policy_for_context(self, info: dict | None = None) -> None:
        context = None if info is None else info.get("context")
        if context is None:
            context = self._get_env_context()

        signature = self._normalize_context(context)
        if signature != self._last_context_signature:
            self._apply_context_to_env(context)
            self._refresh_mdp()
            self._last_context_signature = signature

        if not self.policy_fitted:
            self.update_agent()

    def predict_action(
        self,
        observation: int,
        info: dict | None = None,
        evaluate: bool = False,
    ) -> tuple[int, dict]:
        self._ensure_policy_for_context(info)
        return self.pi[observation], {}
