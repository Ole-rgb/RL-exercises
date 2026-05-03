from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rl_exercises.environments import MarsRover

State = int


class StateValueTDLambdaAgent:
    """Tabular state-value TD(lambda) with accumulating eligibility traces."""

    def __init__(self, alpha: float, gamma: float, lambd: float) -> None:
        assert alpha >= 0, "Learning rate has to be non-negative"
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 <= lambd <= 1, "Lambda should be in [0, 1]"

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambd = float(lambd)
        self.V: DefaultDict[State, float] = defaultdict(float)
        self.eligibility: DefaultDict[State, float] = defaultdict(float)

    def update(
        self,
        state: State,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        """Apply one state-value TD(lambda) update."""
        bootstrap = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * bootstrap - self.V[state]

        self.eligibility[state] += 1.0

        for trace_state in list(self.eligibility.keys()):
            self.V[trace_state] += self.alpha * td_error * self.eligibility[trace_state]
            self.eligibility[trace_state] *= self.gamma * self.lambd

        if done:
            self.eligibility.clear()

        return float(td_error)

    def save(self, path: str | Path) -> None:
        """Save the value table using the same npy style as the Q agents."""
        np.save(path, dict(self.V))


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a generated random-walk training-set file."""
    return np.load(Path(path), allow_pickle=True).item()


def make_eval_env(metadata: dict[str, Any], seed: int) -> MarsRover:
    """Recreate the deterministic MarsRover setup used for the dataset."""
    num_states = int(metadata["num_states"])
    return MarsRover(
        transition_probabilities=np.ones((num_states, 2)),
        rewards=list(metadata["rewards"]),
        horizon=int(metadata["horizon"]),
        seed=seed,
    )


def greedy_value_action(
    env: MarsRover,
    values: DefaultDict[State, float],
    state: State,
    gamma: float,
) -> int:
    """Choose the action maximizing one-step reward plus bootstrapped value."""
    scores = []
    for action in range(env.action_space.n):
        next_state = int(env.get_next_state(state, action))
        reward = float(env.rewards[next_state])
        scores.append(reward + gamma * values[next_state])
    return int(np.argmax(scores))


def evaluate_online(
    agent: StateValueTDLambdaAgent,
    metadata: dict[str, Any],
    episodes: int,
    seed: int,
) -> float:
    """Evaluate the greedy value-derived policy online in MarsRover."""
    env = make_eval_env(metadata, seed)
    episode_rewards: list[float] = []

    for _ in range(episodes):
        state, _ = env.reset(options={"start_state": int(metadata["start_state"])})
        done = False
        episode_reward = 0.0

        while not done:
            action = greedy_value_action(env, agent.V, int(state), agent.gamma)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            done = bool(terminated or truncated)

        episode_rewards.append(episode_reward)

    env.close()
    return float(np.mean(episode_rewards))


@hydra.main(config_path="../configs", config_name="base", version_base="1.1")
def train(cfg: DictConfig) -> float:
    """Train state-value TD(lambda) from generated random-walk trajectories."""
    printr(OmegaConf.to_yaml(cfg))

    dataset = load_dataset(cfg.dataset_path)
    metadata = dataset["metadata"]
    training_sets = dataset["training_sets"]

    agent = StateValueTDLambdaAgent(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        lambd=cfg.lambd,
    )

    train_history: dict[str, list[float | int]] = {"steps": [], "train_rewards": []}
    eval_history: dict[str, list[float | int]] = {"eval_steps": [], "eval_rewards": []}

    global_step = 0
    for _ in range(int(cfg.epochs)):
        for training_set in training_sets:
            for trajectory in training_set:
                agent.eligibility.clear()

                for transition in trajectory:
                    state = int(transition["state"])
                    reward = float(transition["reward"])
                    next_state = int(transition["next_state"])
                    done = bool(transition["done"])

                    agent.update(
                        state=state,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )

                    train_history["steps"].append(global_step)
                    train_history["train_rewards"].append(reward)
                    global_step += 1

                    if (
                        int(cfg.eval_every_n_steps) > 0
                        and global_step % int(cfg.eval_every_n_steps) == 0
                    ):
                        eval_reward = evaluate_online(
                            agent=agent,
                            metadata=metadata,
                            episodes=int(cfg.n_eval_episodes),
                            seed=int(cfg.seed),
                        )
                        eval_history["eval_steps"].append(global_step)
                        eval_history["eval_rewards"].append(eval_reward)
                        print(f"step={global_step} eval_reward={eval_reward:.6f}")

    agent.save(Path("model.csv"))
    pd.DataFrame(train_history).to_csv("train_rewards.csv", index=False)
    pd.DataFrame(eval_history).to_csv("eval_rewards.csv", index=False)
    pd.DataFrame(
        {
            "state": list(range(int(metadata["num_states"]))),
            "value": [
                float(agent.V[state]) for state in range(int(metadata["num_states"]))
            ],
        }
    ).to_csv("value_estimates.csv", index=False)

    print(f"dataset_path={cfg.dataset_path}")
    print(f"transitions_seen={global_step}")
    print(f"model_path={Path.cwd() / 'model.csv.npy'}")
    print("Final state values:")
    for state in range(int(metadata["num_states"])):
        print(f"V({state})={agent.V[state]:.6f}")

    return (
        float(eval_history["eval_rewards"][-1]) if eval_history["eval_rewards"] else 0.0
    )


if __name__ == "__main__":
    train()
