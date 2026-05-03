from __future__ import annotations

import argparse
import subprocess
import sys


def parse_float_list(values: str) -> list[float]:
    """Parse comma-separated floats from a CLI argument."""
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hydra sweep for offline state-value TD(lambda)."
    )
    parser.add_argument(
        "--alphas",
        type=parse_float_list,
        default=parse_float_list("0.0,0.1,0.2,0.3,0.4,0.5,0.6"),
        help="Comma-separated alpha values.",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list("0.3,0.8,0.0,1.0"),
        help="Comma-separated lambda values.",
    )
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every-n-steps", type=int, default=10)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional dataset path override passed to Hydra.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_runs = len(args.alphas) * len(args.lambdas)
    run_index = 0

    for alpha in args.alphas:
        for lambd in args.lambdas:
            run_index += 1
            hydra_overrides = [
                "agent=state_value_tdlambda",
                f"alpha={alpha}",
                f"lambd={lambd}",
                f"gamma={args.gamma}",
                f"epochs={args.epochs}",
                f"seed={args.seed}",
                f"eval_every_n_steps={args.eval_every_n_steps}",
                f"n_eval_episodes={args.n_eval_episodes}",
            ]
            if args.dataset_path is not None:
                hydra_overrides.append(f"dataset_path={args.dataset_path}")

            command = [
                sys.executable,
                "-m",
                "rl_exercises.week_3.train_level_3",
                *hydra_overrides,
            ]

            print(
                f"[{run_index}/{total_runs}] alpha={alpha} lambda={lambd}",
                flush=True,
            )
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
