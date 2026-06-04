"""Proportional prioritized experience replay utilities."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from rl_exercises.agent import AbstractBuffer

Batch: TypeAlias = dict[str, np.ndarray]


class SumTree:
    """Binary sum tree for efficient proportional priority sampling."""

    def __init__(self, capacity: int) -> None:
        """Initialize the tree for a fixed number of leaves."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total_priority(self) -> float:
        """Return the total priority mass stored in the tree."""
        return float(self.tree[0])

    def update(self, data_index: int, priority: float) -> None:
        """Set the stored priority for one data index and propagate the delta."""
        if not 0 <= data_index < self.capacity:
            raise IndexError(
                f"data_index must be in [0, {self.capacity}), got {data_index}"
            )
        if not np.isfinite(priority):
            raise ValueError("priority must be finite")
        if priority < 0.0:
            raise ValueError("priority must be non-negative")

        tree_index = data_index + self.capacity - 1
        delta = float(priority) - float(self.tree[tree_index])
        self.tree[tree_index] = float(priority)

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += delta

    def get(self, cumulative_sum: float) -> tuple[int, float]:
        """Return the leaf selected by the cumulative priority value."""
        total = self.total_priority()
        if total <= 0.0:
            raise ValueError("cannot sample from an empty or zero-priority tree")
        if not np.isfinite(cumulative_sum):
            raise ValueError("cumulative_sum must be finite")

        value = min(max(float(cumulative_sum), 0.0), np.nextafter(total, 0.0))
        tree_index = 0

        while tree_index < self.capacity - 1:
            left = 2 * tree_index + 1
            right = left + 1
            left_sum = self.tree[left]
            if value < left_sum:
                tree_index = left
            else:
                value -= left_sum
                tree_index = right

        data_index = tree_index - (self.capacity - 1)
        return data_index, float(self.tree[tree_index])


class PrioritizedReplayBuffer(AbstractBuffer):
    """Replay buffer with proportional prioritized sampling."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize the replay buffer."""
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.capacity = capacity
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.tree = SumTree(capacity)
        self.states: list[np.ndarray | None] = [None] * capacity
        self.actions: list[int | float | None] = [None] * capacity
        self.rewards: list[float | None] = [None] * capacity
        self.next_states: list[np.ndarray | None] = [None] * capacity
        self.dones: list[bool | None] = [None] * capacity

        self.size = 0
        self.next_idx = 0
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Add a transition, overwriting old data once capacity is reached."""
        del info
        self.states[self.next_idx] = np.array(state, copy=True)
        self.actions[self.next_idx] = action
        self.rewards[self.next_idx] = float(reward)
        self.next_states[self.next_idx] = np.array(next_state, copy=True)
        self.dones[self.next_idx] = bool(done)

        stored_priority = self.max_priority**self.alpha
        self.tree.update(self.next_idx, stored_priority)

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 32) -> tuple[Batch, np.ndarray, np.ndarray]:
        """Sample a batch using stratified proportional prioritization."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        if batch_size > self.size:
            raise ValueError(
                f"batch_size ({batch_size}) cannot exceed current buffer size ({self.size})"
            )

        total_priority = self.tree.total_priority()
        if total_priority <= 0.0:
            raise ValueError("cannot sample when total priority is zero")

        segment = total_priority / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            low = i * segment
            high = (i + 1) * segment
            sample = self.rng.uniform(low, high)
            index, priority = self.tree.get(sample)
            indices[i] = index
            priorities[i] = priority

        batch: Batch = {
            "states": np.stack([self._require_state(idx) for idx in indices], axis=0),
            "actions": np.asarray(
                [self._require_action(idx) for idx in indices],
            ),
            "rewards": np.asarray(
                [self._require_reward(idx) for idx in indices],
                dtype=np.float32,
            ),
            "next_states": np.stack(
                [self._require_next_state(idx) for idx in indices],
                axis=0,
            ),
            "dones": np.asarray(
                [self._require_done(idx) for idx in indices],
                dtype=bool,
            ),
        }

        probs = priorities / total_priority
        weights = (self.size * probs) ** (-self.beta)
        weights /= weights.max()

        return batch, indices, weights.astype(np.float32)

    def update_priorities(
        self, indices: np.ndarray | list[int], td_errors: np.ndarray | list[float]
    ) -> None:
        """Update priorities from temporal-difference errors."""
        index_array = np.asarray(indices, dtype=np.int64)
        error_array = np.asarray(td_errors, dtype=np.float64)

        if index_array.shape != error_array.shape:
            raise ValueError("indices and td_errors must have the same shape")
        if index_array.ndim != 1:
            raise ValueError("indices and td_errors must be one-dimensional")
        if index_array.size == 0:
            return
        if not np.all(np.isfinite(error_array)):
            raise ValueError("td_errors must be finite")
        if np.any(index_array < 0) or np.any(index_array >= self.size):
            raise IndexError(
                f"indices must be in [0, {self.size}), got {index_array.tolist()}"
            )

        raw_priorities = np.abs(error_array) + self.eps
        self.max_priority = max(self.max_priority, float(raw_priorities.max()))

        for index, raw_priority in zip(index_array, raw_priorities):
            self.tree.update(int(index), float(raw_priority**self.alpha))

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size

    def _require_state(self, index: int) -> np.ndarray:
        state = self.states[index]
        if state is None:
            raise RuntimeError(f"missing state for index {index}")
        return state

    def _require_action(self, index: int) -> int | float:
        action = self.actions[index]
        if action is None:
            raise RuntimeError(f"missing action for index {index}")
        return action

    def _require_reward(self, index: int) -> float:
        reward = self.rewards[index]
        if reward is None:
            raise RuntimeError(f"missing reward for index {index}")
        return reward

    def _require_next_state(self, index: int) -> np.ndarray:
        next_state = self.next_states[index]
        if next_state is None:
            raise RuntimeError(f"missing next_state for index {index}")
        return next_state

    def _require_done(self, index: int) -> bool:
        done = self.dones[index]
        if done is None:
            raise RuntimeError(f"missing done flag for index {index}")
        return done


if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(
        capacity=8,
        alpha=0.6,
        beta=0.4,
        eps=1e-6,
    )

    for i in range(8):
        state = np.array([i, i + 1], dtype=np.float32)
        next_state = state + 0.5
        buffer.add(
            state,
            action=i % 2,
            reward=float(i),
            next_state=next_state,
            done=False,
        )

    batch, indices, weights = buffer.sample(batch_size=4)
    print("Sampled indices:", indices)
    print("Weights:", weights)

    td_errors = np.linspace(0.1, 1.0, num=indices.shape[0], dtype=np.float32)
    buffer.update_priorities(indices, td_errors)
    print("Updated priorities for sampled transitions.")
