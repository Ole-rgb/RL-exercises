from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR_RE = re.compile(
    r"seed_(?P<seed>\d+)\.gamma_(?P<gamma>[0-9.]+)"
    r"\.alpha_(?P<alpha>[0-9.]+)\.lambda_(?P<lambd>[0-9.]+)"
    r"\.epochs_(?P<epochs>\d+)$"
)

IDEAL_VALUES = {
    1: 1.0 / 6.0,
    2: 2.0 / 6.0,
    3: 3.0 / 6.0,
    4: 4.0 / 6.0,
    5: 5.0 / 6.0,
}


def compute_rms(value_estimates: pd.DataFrame) -> float:
    """Compute RMS error over non-terminal states 1 through 5."""
    values = value_estimates.set_index("state")["value"]
    squared_errors = [
        (float(values.loc[state]) - ideal_value) ** 2
        for state, ideal_value in IDEAL_VALUES.items()
    ]
    return float(np.sqrt(np.mean(squared_errors)))


def collect_rms(results_dir: Path) -> pd.DataFrame:
    """Collect RMS errors from all state-value TD(lambda) run directories."""
    rows = []

    for value_path in sorted(results_dir.glob("*/value_estimates.csv")):
        match = RUN_DIR_RE.match(value_path.parent.name)
        if match is None:
            continue

        value_estimates = pd.read_csv(value_path)
        rows.append(
            {
                "seed": int(match.group("seed")),
                "gamma": float(match.group("gamma")),
                "alpha": float(match.group("alpha")),
                "lambda": float(match.group("lambd")),
                "epochs": int(match.group("epochs")),
                "rms_error": compute_rms(value_estimates),
            }
        )

    return pd.DataFrame(rows).sort_values(["lambda", "alpha", "seed"])


def plot_rms(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot average RMS error vs alpha for each lambda value."""
    average = (
        summary.groupby(["lambda", "alpha"], as_index=False)["rms_error"]
        .mean()
        .sort_values(["lambda", "alpha"])
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lambd, group in average.groupby("lambda"):
        ax.plot(
            group["alpha"],
            group["rms_error"],
            marker="o",
            linewidth=2,
            label=f"lambda={lambd:g}",
        )

    ax.set_xlabel("alpha")
    ax.set_ylabel("RMS error")
    ax.set_title("State-value TD(lambda) RMS error")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Trace decay")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RMS error for state-value TD(lambda) sweeps."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "state_value_tdlambda" / "MarsRover",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("results")
        / "state_value_tdlambda"
        / "MarsRover"
        / "state_value_tdlambda_rms_summary.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("rl_exercises")
        / "week_3"
        / "plots"
        / "state_value_tdlambda_rms_by_alpha.png",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.3,0.8,1.0",
        help="Comma-separated lambda values to include in the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = collect_rms(args.results_dir)
    lambdas = [float(value.strip()) for value in args.lambdas.split(",")]
    summary = summary[summary["lambda"].isin(lambdas)]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_path, index=False)
    plot_rms(summary, args.output_path)

    print(f"wrote summary to {args.summary_path}")
    print(f"wrote plot to {args.output_path}")


if __name__ == "__main__":
    main()
