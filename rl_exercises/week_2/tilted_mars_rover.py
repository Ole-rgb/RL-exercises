from __future__ import annotations

import numpy as np
from rl_exercises.environments import MarsRover


class TiltedMarsRover(MarsRover):
    """
    MarsRover variant with a global terrain tilt.

    The tilt does not move the rover directly. Instead, it biases how likely
    the rover is to successfully execute a requested action:

    - positive tilt: downhill to the right, so moving right gets easier
    - negative tilt: downhill to the left, so moving left gets easier

    The bias is applied to the base action-follow probabilities ``base_P``.
    For a tilt angle of zero, the environment behaves exactly like the base
    environment. For non-zero tilt, a signed bias is computed as

    ``bias = max_tilt_bias * (tilt_angle / max_tilt_angle)``.

    This means:

    - a positive bias decreases ``P[s, left]`` and increases ``P[s, right]``
    - a negative bias increases ``P[s, left]`` and decreases ``P[s, right]``

    The updated probabilities are clipped to ``probability_bounds`` so the
    environment remains stochastic and probabilities stay valid.
    """

    def __init__(
        self,
        transition_probabilities: np.ndarray = np.ones((5, 2)),
        rewards: list[float] = [1, 0, 0, 0, 10],
        horizon: int = 10,
        seed: int | None = None,
        tilt_angle: float = 0.0,
        max_tilt_angle: float = 30.0,
        max_tilt_bias: float = 0.15,
        probability_bounds: tuple[float, float] = (0.05, 0.95),
    ) -> None:
        """
        Parameters
        ----------
        transition_probabilities : np.ndarray, optional
            Base action-follow probabilities ``P[s, a]`` before tilt is
            applied. ``P[s, a]`` is the probability that action ``a`` is
            executed as intended in state ``s``.
        rewards : list[float], optional
            Reward assigned to each rover position.
        horizon : int, optional
            Maximum number of steps per episode.
        seed : int | None, optional
            Random seed for the environment RNG.
        tilt_angle : float, optional
            Global terrain slope in degrees. Positive angles tilt the terrain
            to the right, negative angles tilt it to the left.
        max_tilt_angle : float, optional
            Largest magnitude of admissible tilt. Used both for validation and
            for normalizing the tilt effect.
        max_tilt_bias : float, optional
            Maximum probability shift applied at ``|tilt_angle| ==
            max_tilt_angle``.
        probability_bounds : tuple[float, float], optional
            Lower and upper clipping bounds for the tilt-adjusted action
            follow probabilities.
        """
        super().__init__(
            transition_probabilities=transition_probabilities,
            rewards=rewards,
            horizon=horizon,
            seed=seed,
        )

        self.base_P = np.array(self.P, dtype=float, copy=True)
        self.max_tilt_angle = float(max_tilt_angle)
        self.max_tilt_bias = float(max_tilt_bias)
        self.probability_bounds = probability_bounds

        self._validate_tilt_configuration()
        self.set_tilt_angle(tilt_angle)

    def _validate_tilt_configuration(self) -> None:
        if self.base_P.ndim != 2 or self.base_P.shape[1] != self.action_space.n:
            raise ValueError(
                "transition_probabilities must have shape (num_states, 2) for TiltedMarsRover."
            )

        if self.max_tilt_angle <= 0:
            raise ValueError("max_tilt_angle must be positive.")

        if self.max_tilt_bias < 0:
            raise ValueError("max_tilt_bias must be non-negative.")

        p_min, p_max = self.probability_bounds
        if not (0.0 <= p_min <= p_max <= 1.0):
            raise ValueError("probability_bounds must satisfy 0 <= min <= max <= 1.")

    def _validate_tilt_angle(self, tilt_angle: float) -> float:
        tilt_angle = float(tilt_angle)
        if abs(tilt_angle) > self.max_tilt_angle:
            raise ValueError(
                f"tilt_angle must be within [-{self.max_tilt_angle}, {self.max_tilt_angle}] degrees."
            )
        return tilt_angle

    def _compute_tilted_probabilities(self, tilt_angle: float) -> np.ndarray:
        """
        Apply the tilt bias to the base action-follow probabilities.

        The bias is linear in the normalized tilt angle:

        ``bias = max_tilt_bias * (tilt_angle / max_tilt_angle)``

        The left-action follow probability is shifted by ``-bias`` and the
        right-action follow probability by ``+bias``. This makes downhill
        movement easier and uphill movement harder while keeping the action
        semantics unchanged.
        """
        tilt_angle = self._validate_tilt_angle(tilt_angle)
        if self.max_tilt_angle == 0:
            return np.array(self.base_P, copy=True)

        bias = self.max_tilt_bias * (tilt_angle / self.max_tilt_angle)
        p_min, p_max = self.probability_bounds

        tilted_probabilities = np.array(self.base_P, dtype=float, copy=True)
        # Positive bias means "downhill to the right", so right becomes easier
        # to execute and left becomes harder. Negative bias does the opposite.
        tilted_probabilities[:, 0] = np.clip(self.base_P[:, 0] - bias, p_min, p_max)
        tilted_probabilities[:, 1] = np.clip(self.base_P[:, 1] + bias, p_min, p_max)
        return tilted_probabilities

    def set_tilt_angle(self, tilt_angle: float) -> None:
        """
        Update the global tilt and rebuild the transition model.

        Changing the tilt changes the effective action-follow probabilities,
        which in turn changes the MDP transition matrix used by planning
        algorithms.
        """
        self.tilt_angle = self._validate_tilt_angle(tilt_angle)
        self.P = self._compute_tilted_probabilities(self.tilt_angle)
        self.transition_matrix = self.T = self.get_transition_matrix()

    def get_transition_matrix(
        self,
        S: np.ndarray | None = None,
        A: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct a stochastic transition matrix T[s, a, s'].

        T[s, a, s'] includes the probability of both outcomes:
        following the intended action with probability P[s, a], and
        executing the opposite action with probability 1 - P[s, a].
        """
        if S is None or A is None or P is None:
            S, A, P = self.states, self.actions, self.P

        nS, nA = len(S), len(A)
        T = np.zeros((nS, nA, nS), dtype=float)

        for s in S:
            for a in A:
                p_follow = float(P[s, a])
                s_follow = self.get_next_state(s, a)
                s_flip = self.get_next_state(s, 1 - a)

                T[s, a, s_follow] += p_follow
                T[s, a, s_flip] += 1.0 - p_follow

        return T
