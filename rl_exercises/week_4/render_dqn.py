"""
Render a saved DQN checkpoint.
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from rl_exercises.week_4.dqn import DQNAgent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render a saved DQN agent.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/dqn/CartPole-v1/seed_0/model.pt"),
        help="Path to the saved DQN checkpoint.",
    )
    parser.add_argument(
        "--env-name",
        default="CartPole-v1",
        help="Gymnasium environment name.",
    )
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array"],
        help="Gymnasium render mode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment and agent seed.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay after each environment step, in seconds.",
    )
    return parser.parse_args()


def build_agent(env: gym.Env, seed: int) -> DQNAgent:
    """Build a DQNAgent with the current CartPole config defaults."""
    return DQNAgent(
        env,
        buffer_capacity=10000,
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        target_update_freq=1000,
        seed=seed,
    )


def main() -> None:
    """Render one or more greedy DQN episodes."""
    args = parse_args()
    env = gym.make(args.env_name, render_mode=args.render_mode)
    agent = build_agent(env, args.seed)
    agent.load(str(args.model_path))

    for episode in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + episode - 1)
        done = False
        total_reward = 0.0

        if args.render_mode == "rgb_array":
            env.render()

        while not done:
            action, _ = agent.predict_action(obs, info, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if args.render_mode == "rgb_array":
                env.render()
            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"Episode {episode} reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
