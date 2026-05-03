from __future__ import annotations

from typing import Any, DefaultDict, Literal

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3 import EpsilonGreedyPolicy

State = Any


class TDAgent(AbstractAgent):
    """SARSA and Q-Learning agent"""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        algorithm: Literal["sarsa", "qlearning"] = "sarsa",
    ) -> None:
        """Initialize the TD agent

        Parameters
        ----------
        env : gym.Env
            Environment for the agent
        alpha : float, optional
            Learning Rate, by default 0.5
        gamma : float, optional
            Discount Factor , by default 1.0
        algorithm : Literal["sarsa", "qlearning"], optional
            Whether to use SARSA (on-policy) or Q-Learning (off-policy), by default "sarsa"
        """
        # Check hyperparameter boundaries
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert algorithm in [
            "sarsa",
            "qlearning",
        ], "algorithm must be 'sarsa' or 'qlearning'"

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.algorithm = algorithm

        # number of actions → used by Q’s default factory
        self.n_actions = env.action_space.n

        # Build Q so that unseen states map to zero‐vectors
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

        self.policy = policy

    def predict_action(
        self, state: np.array, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore # noqa
        """Predict the action for a given state"""
        return self.policy(self.Q, state, evaluate=evaluate), info

    def save(self, path: str) -> Any:  # type: ignore
        """Save the Q table

        Parameters
        ----------
        path :
            Path to save the Q table

        """
        np.save(path, dict(self.Q))  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table

        Parameters
        ----------
        path :
            Path to saved the Q table

        """
        loaded_q = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float),
            loaded_q,
        )

    def update_agent(self, batch) -> float:  # type: ignore
        """Unpack a batch from SimpleBuffer and route to the appropriate TD update.

        Parameters
        ----------
        batch : list
            List of (state, action, reward, next_state, done, info) tuples

        Returns
        -------
        float
            New Q value for the state action pair
        """
        state, action, reward, next_state, done, _ = batch[0]
        if self.algorithm == "sarsa":
            # TODO: Get the next action for the lookahead in SARSA using the policy of this agent.
            next_action = self.predict_action(next_state, info={}, evaluate=True)[0]  # type: ignore
            return self.SARSA(state, action, reward, next_state, next_action, done)
        else:
            return self.Q_Learning(state, action, reward, next_state, done)

    def SARSA(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform a SARSA update (on-policy)
        Q[s,a] ← Q[s,a] + alpha*[r + gamma*Q(s',a') - Q(s,a)]

        Parameters
        ----------
        state : State
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : State
            Next state
        next_action : int
            Next action for lookahead
        done : bool
            Whether the episode is finished

        Returns
        -------
        float
            New Q value for the state action pair
        """

        # SARSA update rule
        # DONE: Implement the SARSA update rule here.
        # Use a value of 0. for terminal states and
        # update the new Q value in the Q table of this class.
        # Return the new Q value --currently always returns 0.0

        # NOTE: if the next state is terminal, the Q value for the next state-action pair is considered to be 0
        # so we only care about the immediate reward and the current Q value for the state-action pair
        td_error = (
            reward
            + self.gamma * self.Q[next_state][next_action] * (1 - done)
            - self.Q[state][action]
        )
        self.Q[state][action] += self.alpha * td_error
        return self.Q[state][action]

    def Q_Learning(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        """Perform a Q-Learning update (off-policy)
        Q[s,a] ← Q[s,a] + alpha*[r + gamma*max(Q(s',·)) - Q(s,a)]

        Parameters
        ----------
        state : State
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : State
            Next state
        done : bool
            Whether the episode is finished

        Returns
        -------
        float
            New Q value for the state action pair
        """

        # Q learning update rule
        # DONE: Implement the Q-Learning update rule here.
        update = (
            reward
            + self.gamma * np.max(self.Q[next_state]) * (1 - done)
            - self.Q[state][action]
        )
        self.Q[state][action] += self.alpha * update
        return self.Q[state][action]


class TDLambdaAgent(TDAgent):
    """Tabular SARSA(lambda) agent with accumulating eligibility traces."""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        lambd: float = 1.0,
    ) -> None:
        """Initialize the TD(lambda) agent.

        Parameters
        ----------
        env : gym.Env
            Environment for the agent.
        policy : EpsilonGreedyPolicy
            Policy used for action selection.
        alpha : float, optional
            Learning rate, by default 0.5.
        gamma : float, optional
            Discount factor, by default 1.0.
        lambd : float, optional
            Trace decay parameter, by default 1.0.
        """
        assert 0 <= lambd <= 1, "Lambda should be in [0, 1]"
        super().__init__(
            env=env,
            policy=policy,
            alpha=alpha,
            gamma=gamma,
            algorithm="sarsa",
        )
        self.lambd = lambd
        self.eligibility: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def update_agent(self, batch) -> float:  # type: ignore
        """Apply one SARSA(lambda) update from the latest transition."""
        state, action, reward, next_state, done, _ = batch[0]
        next_action = self.predict_action(next_state, info={}, evaluate=False)[0]
        return self.SARSA_lambda(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            next_action=int(next_action),
            done=bool(done),
        )

    def SARSA_lambda(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform one accumulating-trace SARSA(lambda) update."""
        td_error = (
            reward
            + self.gamma * self.Q[next_state][next_action] * (1 - done)
            - self.Q[state][action]
        )

        self.eligibility[state][action] += 1.0

        for trace_state in list(self.eligibility.keys()):
            self.Q[trace_state] += self.alpha * td_error * self.eligibility[trace_state]
            self.eligibility[trace_state] *= self.gamma * self.lambd

        if done:
            self.eligibility.clear()

        return self.Q[state][action]
