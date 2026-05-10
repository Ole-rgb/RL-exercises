import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [0, 21, 42]

# Experimental setup for level 1
#    Each entry is a dict with:
#       label      : name shown in the plot legend
#       base_dir   : where Hydra will write results
#       overrides  : list of Hydra override strings on top of your dqn.yaml defaults
WIDTH_EXPS = [
    {
        "label": "narrow (hidden=32)",
        "base_dir": "results/dqn/exp_l1/narrow",
        "overrides": ["network.hidden_dim=32"],
    },
    {
        "label": "baseline (hidden=64)",
        "base_dir": "results/dqn/exp_l1/baseline",
        "overrides": [],
    },
    {
        "label": "wide (hidden=256)",
        "base_dir": "results/dqn/exp/wide",
        "overrides": ["network.hidden_dim=256"],
    },
]
DEPTH_EXPS = [
    {
        "label": "shallow (depth=1)",
        "base_dir": "results/dqn/exp_l1/shallow",
        "overrides": ["network.num_linear_layers=1"],
    },
    {
        "label": "baseline (depth=2)",
        "base_dir": "results/dqn/exp_l1/baseline",
        "overrides": [],
    },
    {
        "label": "deep (depth=4)",
        "base_dir": "results/dqn/exp_l1/deep",
        "overrides": ["network.num_linear_layers=4"],
    },
]
BUFFER_EXPS = [
    {
        "label": "tiny buffer (500)",
        "base_dir": "results/dqn/exp_l1/buf500",
        "overrides": ["agent.buffer_capacity=500"],
    },
    {
        "label": "baseline (10000)",
        "base_dir": "results/dqn/exp_l1/baseline",
        "overrides": [],
    },
    {
        "label": "large buffer (50000)",
        "base_dir": "results/dqn/exp_l1/buf50k",
        "overrides": ["agent.buffer_capacity=50000"],
    },
]
BATCH_EXPS = [
    {
        "label": "small batch (8)",
        "base_dir": "results/dqn/exp_l1/batch8",
        "overrides": ["agent.batch_size=8"],
    },
    {
        "label": "baseline (batch=32)",
        "base_dir": "results/dqn/exp_l1/baseline",
        "overrides": [],
    },
    {
        "label": "large batch (256)",
        "base_dir": "results/dqn/exp_l1/batch256",
        "overrides": ["agent.batch_size=256"],
    },
]
# All unique experiments to actually run
ALL_EXPERIMENTS = WIDTH_EXPS + [
    e
    for e in DEPTH_EXPS + BUFFER_EXPS + BATCH_EXPS
    if e["base_dir"] != "results/dqn/exp_l1/baseline"
]


def run_experiment(label: str, base_dir: str, overrides: list, seed: int):
    """
    Run a single DQN training experiment.

    Calls dqn.py with the given overrides and a fixed output directory.

    Parameters
    ----------
    label : str
        Name used in log output and plotting.
    base_dir : str
        Root directory for the given configuration.
    overrides : list of str
        Overrides applied on top of the dqn.yaml defaults.
    seed : int
        Random seed passed to the agent.

    """

    output_dir = os.path.join(base_dir, f"seed_{seed}")

    all_overrides = overrides + [
        f"seed={seed}",
        f"agent.seed={seed}",
        f"hydra.run.dir={output_dir}",
    ]

    cmd = [
        sys.executable,
        os.path.join("rl_exercises", "week_4", "dqn.py"),
    ] + all_overrides
    print(f"\n{'=' * 60}")
    print(f"\nRunning: {label} | seed={seed}")
    print(f"Output:  {output_dir}")
    print("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise ValueError(
            f"[ERROR] Experiment '{label}' failed with exit code {result.returncode}"
        )


def load_eval_csv(directory: str) -> pd.DataFrame:
    """Load eval_rewards.csv"""
    direct = os.path.join(directory, "eval_rewards.csv")
    if not os.path.exists(direct):
        raise ValueError("eval_rewards.csv not found")
    return pd.read_csv(direct)


def plot_all(experiments: list[dict], output_path: str, title: str):
    """
    Load multi-seed results and produce a mean ± std training curve plot.

    Parameters
    ----------
    experiments : list of dict
        Each dict must contain:
            "label"    : str   — legend label
            "base_dir" : str   — root directory containing seed_N subdirs
            "overrides": list  — (unused here, present for structural consistency)
    output_path : str
        File path where the PNG is saved.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for exp in experiments:
        all_rewards = []
        frames = None

        for seed in SEEDS:
            seed_dir = os.path.join(exp["base_dir"], f"seed_{seed}")
            csv_path = os.path.join(seed_dir, "eval_rewards.csv")
            df = pd.read_csv(csv_path)
            frames = df["eval_steps"].tolist()
            all_rewards.append(df["eval_rewards"].tolist())

        rewards_arr = np.array(all_rewards)  # shape: (n_seeds, n_evals)
        mean = rewards_arr.mean(axis=0)
        std = rewards_arr.std(axis=0)

        (line,) = ax.plot(frames, mean, linewidth=2, label=exp["label"])
        ax.fill_between(
            frames, mean - std, mean + std, alpha=0.1, color=line.get_color()
        )

    ax.set_xlabel("Frames (environment steps)")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title(f"{title} (mean ± std over {len(SEEDS)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0, top=510)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # Run all unique configs across all seeds (baseline runs once)
    seen_dirs = set()
    for exp in ALL_EXPERIMENTS:
        if exp["base_dir"] in seen_dirs:
            continue
        seen_dirs.add(exp["base_dir"])
        for seed in SEEDS:
            run_experiment(exp["label"], exp["base_dir"], exp["overrides"], seed)

    plot_all(
        WIDTH_EXPS,
        "rl_exercises/week_4/plots/l1/plot_width.png",
        title="Effect of Network Width",
    )
    plot_all(
        DEPTH_EXPS,
        "rl_exercises/week_4/plots/l1/plot_depth.png",
        title="Effect of Network Depth",
    )
    plot_all(
        BUFFER_EXPS,
        "rl_exercises/week_4/plots/l1/plot_buffer.png",
        title="Effect of Replay Buffer Size",
    )
    plot_all(
        BATCH_EXPS,
        "rl_exercises/week_4/plots/l1/plot_batch.png",
        title="Effect of Batch Size",
    )
