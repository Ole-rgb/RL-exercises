from __future__ import annotations

from typing import Any, List

import os

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich import print as printr
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_2.contextual_policy_iteration import ContextualPolicyIteration
from rl_exercises.week_2.contextual_value_iteration import ContextualValueIteration
from rl_exercises.week_2.policy_iteration import PolicyIteration
from rl_exercises.week_2.tilted_mars_rover import TiltedMarsRover
from rl_exercises.week_2.value_iteration import ValueIteration
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm


def _get_base_env(env: gym.Env) -> TiltedMarsRover:
    if hasattr(env, "unwrapped"):
        return env.unwrapped  # type: ignore[return-value]
    return env  # type: ignore[return-value]


def _normalize_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, float]]:
    normalized = []
    for context in contexts:
        normalized.append(
            {
                "tilt_angle": float(context["tilt_angle"]),
                "friction": float(context["friction"]),
            }
        )
    return normalized


def _set_env_context(env: gym.Env, context: dict[str, float]) -> None:
    base_env = _get_base_env(env)
    base_env.set_context(
        tilt_angle=float(context["tilt_angle"]),
        friction=float(context["friction"]),
    )


def _get_context_for_episode(
    contexts: list[dict[str, float]],
    episode_idx: int,
    schedule: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    if not contexts:
        raise ValueError("At least one training context is required.")

    if schedule == "round_robin":
        return contexts[episode_idx % len(contexts)]
    if schedule == "random":
        return contexts[int(rng.integers(0, len(contexts)))]

    raise ValueError(
        f"Unknown context_schedule='{schedule}'. Expected 'round_robin' or 'random'."
    )


@hydra.main("../configs", "base", version_base="1.1")  # type: ignore[misc]
def train(cfg: DictConfig) -> float:
    """
    Train a contextual planning agent on the contextual Mars rover.
    """
    env = make_env(cfg.env_name, cfg.env_kwargs)
    printr(cfg)
    train_contexts = _normalize_contexts(list(cfg.train_contexts))
    validation_contexts = _normalize_contexts(list(cfg.validation_contexts))
    test_contexts = _normalize_contexts(list(cfg.test_contexts))
    schedule = str(cfg.context_schedule)
    rng = np.random.default_rng(cfg.seed)

    if cfg.agent == "contextual_policy_iteration":
        agent = ContextualPolicyIteration(env)
    elif cfg.agent == "contextual_value_iteration":
        agent = ContextualValueIteration(env)
    elif cfg.agent == "non_contextual_policy_iteration":
        agent = PolicyIteration(env)
    elif cfg.agent == "non_contextual_value_iteration":
        agent = ValueIteration(env)
    else:
        raise NotImplementedError(
            "contextual_train_agent.py only supports contextual and non-contextual planning agents."
        )

    buffer_cls = eval(cfg.buffer_cls)
    buffer = buffer_cls(**cfg.buffer_kwargs)
    episode_idx = 0
    _set_env_context(
        env, _get_context_for_episode(train_contexts, episode_idx, schedule, rng)
    )
    state, info = env.reset(seed=cfg.seed)
    current_context = _get_context_for_episode(
        train_contexts, episode_idx, schedule, rng
    )
    train_reward_buffer = {
        "steps": [],
        "train_rewards": [],
        "tilt_angle": [],
        "friction": [],
    }
    eval_reward_buffer = {
        "eval_steps": [],
        "train_context_mean_reward": [],
        "validation_mean_reward": [],
    }

    for step in range(int(cfg.training_steps)):
        action, _ = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)

        buffer.add(state, action, reward, next_state, (truncated or terminated), info)
        train_reward_buffer["steps"].append(step)
        train_reward_buffer["train_rewards"].append(reward)
        train_reward_buffer["tilt_angle"].append(current_context["tilt_angle"])
        train_reward_buffer["friction"].append(current_context["friction"])

        if len(buffer) > cfg.batch_size or (
            cfg.update_after_episode_end and (terminated or truncated)
        ):
            batch = buffer.sample(cfg.batch_size)
            agent.update_agent(batch)

        state = next_state

        if terminated or truncated:
            episode_idx += 1
            current_context = _get_context_for_episode(
                train_contexts, episode_idx, schedule, rng
            )
            _set_env_context(env, current_context)
            state, info = env.reset(seed=cfg.seed)

        if step % cfg.eval_every_n_steps == 0:
            train_performance = evaluate_contexts(
                agent,
                train_contexts,
                cfg.env_name,
                dict(cfg.env_kwargs),
                cfg.n_eval_episodes,
                cfg.seed,
            )
            validation_performance = evaluate_contexts(
                agent,
                validation_contexts,
                cfg.env_name,
                dict(cfg.env_kwargs),
                cfg.n_eval_episodes,
                cfg.seed,
            )
            print(
                f"Eval after {step} steps: train={train_performance:.3f}, "
                f"validation={validation_performance:.3f}"
            )
            eval_reward_buffer["eval_steps"].append(step)
            eval_reward_buffer["train_context_mean_reward"].append(train_performance)
            eval_reward_buffer["validation_mean_reward"].append(validation_performance)

    agent.save(str(os.path.abspath("model.csv")))
    pd.DataFrame(train_reward_buffer).to_csv(
        os.path.abspath("train_rewards.csv"), index=False
    )
    pd.DataFrame(eval_reward_buffer).to_csv(
        os.path.abspath("eval_rewards.csv"), index=False
    )
    final_train = evaluate_contexts(
        agent,
        train_contexts,
        cfg.env_name,
        dict(cfg.env_kwargs),
        cfg.n_eval_episodes,
        cfg.seed,
    )
    final_validation = evaluate_contexts(
        agent,
        validation_contexts,
        cfg.env_name,
        dict(cfg.env_kwargs),
        cfg.n_eval_episodes,
        cfg.seed,
    )
    final_test = evaluate_contexts(
        agent,
        test_contexts,
        cfg.env_name,
        dict(cfg.env_kwargs),
        cfg.n_eval_episodes,
        cfg.seed,
    )
    print(
        f"Final contextual eval: train={final_train:.3f}, "
        f"validation={final_validation:.3f}, test={final_test:.3f}"
    )
    final_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "mean_reward": final_train,
                "num_contexts": len(train_contexts),
            },
            {
                "split": "validation",
                "mean_reward": final_validation,
                "num_contexts": len(validation_contexts),
            },
            {
                "split": "test",
                "mean_reward": final_test,
                "num_contexts": len(test_contexts),
            },
        ]
    )
    final_summary.to_csv(os.path.abspath("final_eval_summary.csv"), index=False)
    final_details = pd.concat(
        [
            evaluate_contexts_detailed(
                agent,
                train_contexts,
                cfg.env_name,
                dict(cfg.env_kwargs),
                cfg.n_eval_episodes,
                cfg.seed,
                split_name="train",
            ),
            evaluate_contexts_detailed(
                agent,
                validation_contexts,
                cfg.env_name,
                dict(cfg.env_kwargs),
                cfg.n_eval_episodes,
                cfg.seed,
                split_name="validation",
            ),
            evaluate_contexts_detailed(
                agent,
                test_contexts,
                cfg.env_name,
                dict(cfg.env_kwargs),
                cfg.n_eval_episodes,
                cfg.seed,
                split_name="test",
            ),
        ],
        ignore_index=True,
    )
    final_details.to_csv(os.path.abspath("final_eval_per_context.csv"), index=False)
    return final_test


def evaluate(
    env: gym.Env,
    agent: AbstractAgent,
    episodes: int = 100,
    seed: int = 0,
    context: dict[str, float] | None = None,
) -> float:
    """
    Evaluate a contextual agent on a contextual environment.
    """
    if context is not None:
        _set_env_context(env, context)

    episode_rewards: List[float] = []
    pbar = tqdm(total=episodes)
    for _ in range(episodes):
        obs, info = env.reset(seed=seed)
        episode_rewards.append(0)
        done = False
        episode_steps = 0
        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)  # type: ignore[arg-type]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards[-1] += reward
            episode_steps += 1
            if terminated or truncated:
                done = True
                pbar.set_postfix(
                    {
                        "episode reward": episode_rewards[-1],
                        "episode step": episode_steps,
                    }
                )
        pbar.update(1)
    env.close()
    return np.mean(episode_rewards)


def evaluate_contexts(
    agent: AbstractAgent,
    contexts: list[dict[str, float]],
    env_name: str,
    env_kwargs: dict[str, Any],
    episodes_per_context: int = 1,
    seed: int = 0,
) -> float:
    """
    Evaluate an agent across a set of contexts and return the mean reward.
    """
    if not contexts:
        raise ValueError("Evaluation contexts must not be empty.")

    rewards = []
    for context in contexts:
        eval_env = make_env(env_name, env_kwargs)
        rewards.append(
            evaluate(
                env=eval_env,
                agent=agent,
                episodes=episodes_per_context,
                seed=seed,
                context=context,
            )
        )
    return float(np.mean(rewards))


def evaluate_contexts_detailed(
    agent: AbstractAgent,
    contexts: list[dict[str, float]],
    env_name: str,
    env_kwargs: dict[str, Any],
    episodes_per_context: int = 1,
    seed: int = 0,
    split_name: str = "eval",
) -> pd.DataFrame:
    """
    Evaluate each context separately and return a table of results.
    """
    rows = []
    for context in contexts:
        mean_reward = evaluate(
            env=make_env(env_name, env_kwargs),
            agent=agent,
            episodes=episodes_per_context,
            seed=seed,
            context=context,
        )
        rows.append(
            {
                "split": split_name,
                "tilt_angle": context["tilt_angle"],
                "friction": context["friction"],
                "mean_reward": mean_reward,
            }
        )
    return pd.DataFrame(rows)


def make_env(env_name: str, env_kwargs: dict = {}) -> gym.Env:
    """
    Build a week 2 contextual environment.
    """
    if env_name != "TiltedMarsRover":
        raise NotImplementedError(
            "contextual_train_agent.py expects env_name='TiltedMarsRover'."
        )

    env = TiltedMarsRover(**env_kwargs)
    return Monitor(env, filename="train")


if __name__ == "__main__":
    train()
