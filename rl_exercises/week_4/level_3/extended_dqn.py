"""Extended DQN agent with configurable replay buffer selection."""

from __future__ import annotations

from typing import Any, Sequence

from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rl_exercises.agent import AbstractBuffer
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.dqn import DQNAgent, evaluate, set_seed
from rl_exercises.week_4.level_3.per import PrioritizedReplayBuffer


class ExtendedDQNAgent(DQNAgent):
    """DQN variant that supports an injected replay buffer implementation."""

    SUPPORTED_MODES = {"normal", "double_dqn"}

    def __init__(
        self,
        env: gym.Env,
        replay_buffer: AbstractBuffer,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        network_cfg: DictConfig = DictConfig(
            {"hidden_dim": 64, "num_linear_layers": 2}
        ),
        seed: int = 0,
        mode: str = "normal",
    ) -> None:
        """Initialize the base DQN stack and replace the replay buffer."""
        super().__init__(
            env=env,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            network_cfg=network_cfg,
            seed=seed,
        )
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}. Expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        self.buffer = replay_buffer
        self.mode = mode

    def _batch_to_tensors(
        self, training_batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert supported batch formats into tensors."""
        if isinstance(training_batch, dict):
            states = torch.tensor(training_batch["states"], dtype=torch.float32)
            actions = torch.tensor(
                training_batch["actions"], dtype=torch.int64
            ).unsqueeze(1)
            rewards = torch.tensor(training_batch["rewards"], dtype=torch.float32)
            next_states = torch.tensor(
                training_batch["next_states"], dtype=torch.float32
            )
            dones = torch.tensor(training_batch["dones"], dtype=torch.float32)
            return states, actions, rewards, next_states, dones

        if not isinstance(training_batch, Sequence):
            raise TypeError(
                "training_batch must be a dict or a sequence of transitions"
            )

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in training_batch:
            if len(transition) == 6:
                state, action, reward, next_state, done, _ = transition
            elif len(transition) == 5:
                state, action, reward, next_state, done = transition
            else:
                raise ValueError(
                    "Each transition must have 5 or 6 elements: "
                    "(state, action, reward, next_state, done[, info])"
                )
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (
            torch.tensor(np.asarray(states), dtype=torch.float32),
            torch.tensor(np.asarray(actions), dtype=torch.int64).unsqueeze(1),
            torch.tensor(np.asarray(rewards), dtype=torch.float32),
            torch.tensor(np.asarray(next_states), dtype=torch.float32),
            torch.tensor(np.asarray(dones), dtype=torch.float32),
        )

    def _compute_bootstrap_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bootstrapped TD target for the configured DQN mode."""
        with torch.no_grad():
            if self.mode == "normal":
                next_q_values = self.target_q(next_states).max(dim=1)[0]
            elif self.mode == "double_dqn":
                next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
                next_q_values = (
                    self.target_q(next_states).gather(1, next_actions).squeeze(1)
                )
            else:
                raise ValueError(f"Unsupported mode {self.mode!r}")

        return rewards + self.gamma * (1 - dones) * next_q_values

    def update_agent(
        self,
        training_batch: Any,
        indices: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> float:
        """Apply one uniform or prioritized replay update using the configured mode."""
        if (indices is None) != (weights is None):
            raise ValueError(
                "indices and weights must either both be provided or both be omitted"
            )

        states, actions, rewards, next_states, dones = self._batch_to_tensors(
            training_batch
        )

        pred = self.q(states).gather(1, actions).squeeze(1)
        target = self._compute_bootstrap_target(rewards, next_states, dones)
        td_errors = target - pred
        if weights is None:
            loss = td_errors.pow(2).mean()
        else:
            importance_weights = torch.tensor(weights, dtype=torch.float32)
            loss = (importance_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if indices is not None:
            if not isinstance(self.buffer, PrioritizedReplayBuffer):
                raise ValueError(
                    "indices and weights are only valid with prioritized replay"
                )
            self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(
        self,
        num_frames: int,
        eval_interval: int = 1000,
        n_eval_episodes: int = 1,
        seed: int = 0,
    ) -> float:
        """Run the DQN training loop with either uniform or prioritized replay."""
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: list[float] = []
        train_reward_buffer = {"steps": [], "train_rewards": []}
        eval_reward_buffer = {"eval_steps": [], "eval_rewards": []}
        final_model_path = Path("model.pt").resolve()
        best_model_path = Path("best_model.pt").resolve()
        best_eval_reward = float("-inf")

        for frame in range(1, num_frames + 1):
            action, _ = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            train_reward_buffer["steps"].append(frame)
            train_reward_buffer["train_rewards"].append(reward)
            state = next_state
            ep_reward += reward

            if len(self.buffer) >= self.batch_size:
                if isinstance(self.buffer, PrioritizedReplayBuffer):
                    batch, indices, weights = self.buffer.sample(self.batch_size)
                    _ = self.update_agent(batch, indices=indices, weights=weights)
                else:
                    batch = self.buffer.sample(self.batch_size)
                    _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                if len(recent_rewards) % 10 == 0:
                    avg = sum(recent_rewards[-eval_interval:]) / min(
                        len(recent_rewards), eval_interval
                    )
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

            if frame % eval_interval == 0:
                eval_env = gym.make(self.env.spec.id)
                eval_performance = evaluate(
                    eval_env,
                    self,
                    episodes=n_eval_episodes,
                    seed=seed,
                )
                print(f"Eval reward after {frame} steps was {eval_performance}.")
                eval_reward_buffer["eval_steps"].append(frame)
                eval_reward_buffer["eval_rewards"].append(eval_performance)
                if eval_performance > best_eval_reward:
                    best_eval_reward = eval_performance
                    self.save(str(best_model_path))

        print("Training complete.")
        self.save(str(final_model_path))
        pd.DataFrame(train_reward_buffer).to_csv("train_rewards.csv", index=False)
        pd.DataFrame(eval_reward_buffer).to_csv("eval_rewards.csv", index=False)

        final_eval_env = gym.make(self.env.spec.id)
        final_eval = evaluate(
            final_eval_env,
            self,
            episodes=n_eval_episodes,
            seed=seed,
        )
        if final_eval > best_eval_reward:
            self.save(str(best_model_path))
        print(f"Final eval reward was: {final_eval}")
        print(f"Best eval reward was: {max(best_eval_reward, final_eval)}")
        return final_eval


def build_buffer(cfg: DictConfig) -> AbstractBuffer:
    """Instantiate the configured replay buffer."""
    if cfg.type == "uniform":
        return ReplayBuffer(capacity=int(cfg.capacity))
    if cfg.type == "prioritized":
        return PrioritizedReplayBuffer(
            capacity=int(cfg.capacity),
            alpha=float(cfg.alpha),
            beta=float(cfg.beta),
            eps=float(cfg.eps),
        )
    raise ValueError(f"Unsupported buffer type: {cfg.type}")


@hydra.main(
    config_path="../../configs/agent/",
    config_name="extended_dqn",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Train the extended DQN agent with the configured replay buffer."""
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    replay_buffer = build_buffer(cfg.buffer)

    agent_kwargs = {
        "replay_buffer": replay_buffer,
        "buffer_capacity": cfg.buffer.capacity,
        "batch_size": cfg.agent.batch_size,
        "lr": cfg.agent.learning_rate,
        "gamma": cfg.agent.gamma,
        "epsilon_start": cfg.agent.epsilon_start,
        "epsilon_final": cfg.agent.epsilon_final,
        "epsilon_decay": cfg.agent.epsilon_decay,
        "target_update_freq": cfg.agent.target_update_freq,
        "seed": cfg.agent.seed,
        "network_cfg": cfg.network,
        "mode": cfg.agent.mode,
    }

    agent = ExtendedDQNAgent(env, **agent_kwargs)
    agent.train(
        cfg.train.num_frames,
        cfg.train.eval_interval,
        cfg.train.n_eval_episodes,
        cfg.seed,
    )


if __name__ == "__main__":
    main()
