from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rl_exercises.agent.buffer import SimpleBuffer
from rl_exercises.environments import MarsRover
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl_exercises.week_3.sarsa_qlearning import TDLambdaAgent


def evaluate(agent: TDLambdaAgent, episodes: int, seed: int) -> float:
    """Evaluate the greedy policy on a fresh MarsRover environment."""
    env = MarsRover(seed=seed)
    episode_rewards: list[float] = []

    for _ in range(episodes):
        state, info = env.reset(seed=seed)
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = agent.predict_action(state, info, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    env.close()
    return float(np.mean(episode_rewards))


def train(args: argparse.Namespace) -> float:
    """Train TD(lambda) on MarsRover with a dedicated loop."""
    env = MarsRover(seed=args.seed)
    policy = EpsilonGreedyPolicy(env, epsilon=args.epsilon, seed=args.seed)
    agent = TDLambdaAgent(
        env=env,
        policy=policy,
        alpha=args.alpha,
        gamma=args.gamma,
        lambd=args.lambd,
    )
    buffer = SimpleBuffer()

    state, info = env.reset(seed=args.seed)
    train_history: dict[str, list[float | int]] = {"step": [], "reward": []}
    eval_history: dict[str, list[float | int]] = {"step": [], "reward": []}

    for step in range(args.training_steps):
        action, info = agent.predict_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, float(reward), next_state, done, info)
        agent.update_agent(buffer.sample())

        train_history["step"].append(step)
        train_history["reward"].append(float(reward))

        state = next_state
        if done:
            state, info = env.reset(seed=args.seed)

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_reward = evaluate(agent, args.eval_episodes, args.seed)
            eval_history["step"].append(step)
            eval_history["reward"].append(eval_reward)
            print(f"step={step} eval_reward={eval_reward:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.output_dir / "tdlambda_q.npy"))
    pd.DataFrame(train_history).to_csv(
        args.output_dir / "train_rewards.csv", index=False
    )
    pd.DataFrame(eval_history).to_csv(args.output_dir / "eval_rewards.csv", index=False)

    final_eval = evaluate(agent, args.eval_episodes, args.seed)
    print(f"final_eval_reward={final_eval:.3f}")
    return final_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TD(lambda) on MarsRover.")
    parser.add_argument("--training-steps", type=int, default=1500)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambd", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "tdlambda" / "MarsRover",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
