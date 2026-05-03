from __future__ import annotations

from typing import Any

import argparse
from pathlib import Path

import numpy as np
from rl_exercises.environments import MarsRover

Step = dict[str, int | float | bool]
Trajectory = list[Step]
TrainingSet = list[Trajectory]


def make_mars_rover(
    num_states: int,
    horizon: int,
    env_seed: int,
) -> MarsRover:
    """Create the deterministic 1D MarsRover random-walk environment."""
    rewards = [0.0] * num_states
    rewards[-1] = 1.0

    return MarsRover(
        transition_probabilities=np.ones((num_states, 2)),
        rewards=rewards,
        horizon=horizon,
        seed=env_seed,
    )


def collect_trajectory(
    env: MarsRover,
    rng: np.random.Generator,
    start_state: int,
) -> Trajectory:
    """Collect one random-walk trajectory until the environment ends the episode."""
    state, _ = env.reset(options={"start_state": start_state})
    done = False
    trajectory: Trajectory = []

    while not done:
        action = int(rng.integers(env.action_space.n))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        trajectory.append(
            {
                "state": int(state),
                "action": action,
                "reward": float(reward),
                "next_state": int(next_state),
                "done": done,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        state = next_state

    return trajectory


def generate_training_sets(args: argparse.Namespace) -> dict[str, Any]:
    """Generate nested random-walk training sets and metadata."""
    env = make_mars_rover(
        num_states=args.num_states,
        horizon=args.horizon,
        env_seed=args.env_seed,
    )
    rng = np.random.default_rng(args.policy_seed)

    training_sets: list[TrainingSet] = []
    for _ in range(args.num_training_sets):
        training_set: TrainingSet = []
        for _ in range(args.walks_per_set):
            training_set.append(
                collect_trajectory(
                    env=env,
                    rng=rng,
                    start_state=args.start_state,
                )
            )
        training_sets.append(training_set)

    env.close()

    rewards = [0.0] * args.num_states
    rewards[-1] = 1.0

    return {
        "metadata": {
            "num_training_sets": args.num_training_sets,
            "walks_per_set": args.walks_per_set,
            "num_states": args.num_states,
            "start_state": args.start_state,
            "horizon": args.horizon,
            "rewards": rewards,
            "env_seed": args.env_seed,
            "policy_seed": args.policy_seed,
            "deterministic_dynamics": True,
        },
        "training_sets": training_sets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random-walk trajectory training sets for MarsRover."
    )
    parser.add_argument("--num-training-sets", type=int, default=100)
    parser.add_argument("--walks-per-set", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--num-states", type=int, default=7)
    parser.add_argument("--start-state", type=int, default=3)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("rl_exercises/week_3/data/random_walk_training_sets.npy"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_training_sets(args)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, dataset, allow_pickle=True)

    metadata = dataset["metadata"]
    print(
        "saved "
        f"{metadata['num_training_sets']} training sets x "
        f"{metadata['walks_per_set']} walks to {args.output_path}"
    )


if __name__ == "__main__":
    main()
