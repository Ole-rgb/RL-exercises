from __future__ import annotations

import argparse
import itertools
import subprocess
import sys


def parse_int_list(values: str) -> list[int]:
    """Parse comma-separated integers from a CLI argument."""
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_float_list(values: str) -> list[float]:
    """Parse comma-separated floats from a CLI argument."""
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    """Parse sweep settings for normal-DQN training via extended_dqn."""
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep for extended_dqn."
    )
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=parse_int_list(
            "0,1,10,100,1000,10000,100000,1000000,10000000,100000000"
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("normal", "double_dqn"),
        default=["normal", "double_dqn"],
        help="DQN update modes to evaluate.",
    )
    parser.add_argument(
        "--buffer-types",
        nargs="+",
        choices=("uniform", "prioritized"),
        default=["uniform", "prioritized"],
        help="Replay buffer types to evaluate.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=parse_int_list("32,64"),
    )
    parser.add_argument(
        "--buffer-capacities",
        type=parse_int_list,
        default=parse_int_list("100,1000,10000"),
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_int_list,
        default=parse_int_list("32,64,128"),
    )
    parser.add_argument(
        "--num-linear-layers",
        type=parse_int_list,
        default=parse_int_list("2,3,4"),
    )
    parser.add_argument(
        "--extra-overrides",
        nargs="*",
        default=[],
        help="Optional extra Hydra overrides, e.g. train.n_eval_episodes=100.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch one extended_dqn run per parameter combination."""
    args = parse_args()
    combinations = list(
        itertools.product(
            args.seeds,
            args.modes,
            args.buffer_types,
            args.batch_sizes,
            args.buffer_capacities,
            args.hidden_dims,
            args.num_linear_layers,
        )
    )

    total_runs = len(combinations)
    print(
        f"Starting extended DQN sweep with {len(args.seeds)} seed(s): {args.seeds}",
        flush=True,
    )
    print(
        f"Total parameter combinations: {total_runs}",
        flush=True,
    )
    print(
        "Configuration space: "
        f"env_name={args.env_name}, "
        f"modes={args.modes}, "
        f"buffer_types={args.buffer_types}, "
        f"batch_sizes={args.batch_sizes}, "
        f"buffer_capacities={args.buffer_capacities}, "
        f"hidden_dims={args.hidden_dims}, "
        f"num_linear_layers={args.num_linear_layers}, "
        f"extra_overrides={args.extra_overrides}",
        flush=True,
    )
    for run_index, combination in enumerate(combinations, start=1):
        seed, mode, buffer_type, batch_size, buffer_capacity, hidden_dim, num_layers = (
            combination
        )
        hydra_overrides = [
            f"env.name={args.env_name}",
            f"seed={seed}",
            "agent.name=extended_dqn",
            f"agent.mode={mode}",
            f"agent.seed={seed}",
            f"agent.batch_size={batch_size}",
            f"buffer.type={buffer_type}",
            f"buffer.capacity={buffer_capacity}",
            f"network.hidden_dim={hidden_dim}",
            f"network.num_linear_layers={num_layers}",
            *args.extra_overrides,
        ]

        command = [
            sys.executable,
            "-m",
            "rl_exercises.week_4.level_3.extended_dqn",
            *hydra_overrides,
        ]

        print(
            f"[{run_index}/{total_runs}] "
            f"seed={seed} mode={mode} buffer_type={buffer_type} batch_size={batch_size} "
            f"buffer_capacity={buffer_capacity} hidden_dim={hidden_dim} "
            f"num_linear_layers={num_layers}",
            flush=True,
        )
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
