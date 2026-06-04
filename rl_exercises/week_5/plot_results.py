"""
Generate comparison plots for REINFORCE (CartPole-v1) vs DDPG (Pendulum-v1).

Reads the CSV files produced by the training runs and saves two PNG files:
  - reinforce_cartpole_returns.png
  - ddpg_pendulum_returns.png

Run from the week_5 directory:
    python plot_results.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Helpers
WEEK5_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(WEEK5_DIR, "results")

def latest_run(agent_name: str, train_file: str, eval_file: str):
    """Return (train_df, eval_df) from the most recent run for an agent."""
    pattern = os.path.join(RESULTS_DIR, agent_name, "seed_0", "*")
    runs = sorted(glob.glob(pattern))
    if not runs:
        raise FileNotFoundError(
            f"No runs found for agent '{agent_name}' under {RESULTS_DIR}"
        )
    run_dir = runs[-1]  # latest timestamp
    train_df = pd.read_csv(os.path.join(run_dir, train_file))
    eval_df = pd.read_csv(os.path.join(run_dir, eval_file))
    print(f"  Loaded {agent_name} from: {run_dir}")
    return train_df, eval_df

def smooth(values, window: int = 10):
    """Simple moving average for readability."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

# REINFORCE plot
def plot_reinforce():
    # Try REINFORCE first, fall back to REINFORCE_v2
    for agent_name in ("REINFORCE", "REINFORCE_v2"):
        try:
            train_df, eval_df = latest_run(agent_name, "train_rewards.csv", "eval_rewards.csv")
            break
        except FileNotFoundError:
            continue
    else:
        print("WARNING: No REINFORCE results found, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("REINFORCE — CartPole-v1", fontsize=13, fontweight="bold")

    # Left: training return per episode (smoothed)
    ax = axes[0]
    raw = train_df["train_rewards"].values
    eps = train_df["episodes"].values
    ax.plot(eps, raw, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    w = min(20, len(raw) // 5)
    if w > 1:
        smoothed = smooth(raw, w)
        ax.plot(eps[w - 1:], smoothed, color="steelblue", linewidth=1.8, label=f"MA-{w}")
    ax.axhline(475, color="green", linestyle="--", linewidth=1, label="solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Training Return")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(True, alpha=0.3)

    # Right: eval mean ± std
    ax = axes[1]
    e_eps = eval_df["episodes"].values
    mean = eval_df["eval_mean_ret"].values
    std = eval_df["eval_std_ret"].values
    ax.plot(e_eps, mean, color="darkorange", linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(e_eps, mean - std, mean + std, alpha=0.2, color="darkorange")
    ax.axhline(475, color="green", linestyle="--", linewidth=1, label="solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Return")
    ax.set_title("Evaluation Return (mean ± std)")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(WEEK5_DIR, "reinforce_cartpole_returns.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# DDPG plot
def plot_ddpg():
    try:
        train_df, eval_df = latest_run("DDPG", "train_rewards_ddpg.csv", "eval_rewards_ddpg.csv")
    except FileNotFoundError:
        print("WARNING: No DDPG results found, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DDPG — Pendulum-v1", fontsize=13, fontweight="bold")

    # Left: training return per episode (smoothed)
    ax = axes[0]
    raw = train_df["train_rewards"].values
    eps = train_df["episodes"].values
    ax.plot(eps, raw, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    w = min(20, len(raw) // 5)
    if w > 1:
        smoothed = smooth(raw, w)
        ax.plot(eps[w - 1:], smoothed, color="steelblue", linewidth=1.8, label=f"MA-{w}")
    ax.axhline(-200, color="green", linestyle="--", linewidth=1, label="target (−200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Training Return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: eval mean ± std
    ax = axes[1]
    e_eps = eval_df["episodes"].values
    mean = eval_df["eval_mean_ret"].values
    std = eval_df["eval_std_ret"].values
    ax.plot(e_eps, mean, color="darkorange", linewidth=1.8, marker="o", markersize=3)
    ax.fill_between(e_eps, mean - std, mean + std, alpha=0.2, color="darkorange")
    ax.axhline(-200, color="green", linestyle="--", linewidth=1, label="target (−200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Return")
    ax.set_title("Evaluation Return (mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(WEEK5_DIR, "ddpg_pendulum_returns.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# Combined comparison plot (eval curves side by side)
def plot_comparison():
    """One figure with both eval curves for easy side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("REINFORCE vs DDPG — Evaluation Returns", fontsize=13, fontweight="bold")

    # REINFORCE
    ax = axes[0]
    loaded = False
    for agent_name in ("REINFORCE", "REINFORCE_v2"):
        try:
            _, eval_df = latest_run(agent_name, "train_rewards.csv", "eval_rewards.csv")
            loaded = True
            break
        except FileNotFoundError:
            continue
    if loaded:
        e_eps = eval_df["episodes"].values
        mean = eval_df["eval_mean_ret"].values
        std = eval_df["eval_std_ret"].values
        ax.plot(e_eps, mean, color="steelblue", linewidth=1.8, marker="o", markersize=3)
        ax.fill_between(e_eps, mean - std, mean + std, alpha=0.2, color="steelblue")
        ax.axhline(475, color="green", linestyle="--", linewidth=1, label="solved (475)")
        ax.set_title("REINFORCE — CartPole-v1")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # DDPG
    ax = axes[1]
    try:
        _, eval_df = latest_run("DDPG", "train_rewards_ddpg.csv", "eval_rewards_ddpg.csv")
        e_eps = eval_df["episodes"].values
        mean = eval_df["eval_mean_ret"].values
        std = eval_df["eval_std_ret"].values
        ax.plot(e_eps, mean, color="darkorange", linewidth=1.8, marker="o", markersize=3)
        ax.fill_between(e_eps, mean - std, mean + std, alpha=0.2, color="darkorange")
        ax.axhline(-200, color="green", linestyle="--", linewidth=1, label="target (−200)")
        ax.set_title("DDPG — Pendulum-v1")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    except FileNotFoundError:
        pass

    plt.tight_layout()
    out = os.path.join(WEEK5_DIR, "comparison_returns.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

# Main
if __name__ == "__main__":
    print("Generating plots...")
    plot_reinforce()
    plot_ddpg()
    plot_comparison()
    print("Done.")
