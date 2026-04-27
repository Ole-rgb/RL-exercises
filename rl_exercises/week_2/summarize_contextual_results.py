from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "contextual_analysis"


def _parse_run_metadata(summary_path: Path) -> dict[str, str]:
    rel = summary_path.relative_to(RESULTS_DIR)
    parts = rel.parts

    if len(parts) >= 5:
        agent_name = parts[0]
        env_name = parts[1]
        experiment_name = parts[2]
        seed = parts[3]
    elif len(parts) == 4:
        agent_name = parts[0]
        env_name = parts[1]
        experiment_name = "legacy"
        seed = parts[2]
    else:
        raise ValueError(f"Unexpected results path layout: {summary_path}")

    return {
        "agent_name": agent_name,
        "env_name": env_name,
        "experiment_name": experiment_name,
        "seed": seed,
    }


def collect_summary_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    per_context_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []

    for summary_path in sorted(RESULTS_DIR.glob("**/final_eval_summary.csv")):
        metadata = _parse_run_metadata(summary_path)

        summary_df = pd.read_csv(summary_path)
        for key, value in metadata.items():
            summary_df[key] = value
        summary_frames.append(summary_df)

        per_context_path = summary_path.with_name("final_eval_per_context.csv")
        if per_context_path.exists():
            per_context_df = pd.read_csv(per_context_path)
            for key, value in metadata.items():
                per_context_df[key] = value
            per_context_frames.append(per_context_df)

        eval_path = summary_path.with_name("eval_rewards.csv")
        if eval_path.exists():
            eval_df = pd.read_csv(eval_path)
            for key, value in metadata.items():
                eval_df[key] = value
            eval_frames.append(eval_df)

    if not summary_frames:
        raise FileNotFoundError("No contextual result files found under results/.")

    summary = pd.concat(summary_frames, ignore_index=True)
    per_context = (
        pd.concat(per_context_frames, ignore_index=True)
        if per_context_frames
        else pd.DataFrame()
    )
    eval_history = (
        pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    )
    return summary, per_context, eval_history


def save_csv_outputs(
    summary: pd.DataFrame, per_context: pd.DataFrame, eval_history: pd.DataFrame
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary.to_csv(OUTPUT_DIR / "contextual_summary_by_split.csv", index=False)

    summary_pivot = summary.pivot_table(
        index=["agent_name", "env_name", "experiment_name", "seed"],
        columns="split",
        values="mean_reward",
    ).reset_index()
    summary_pivot.to_csv(OUTPUT_DIR / "contextual_summary_wide.csv", index=False)

    test_only = summary[summary["split"] == "test"].copy()
    test_only.to_csv(OUTPUT_DIR / "contextual_test_summary.csv", index=False)

    if not per_context.empty:
        per_context.to_csv(OUTPUT_DIR / "contextual_per_context.csv", index=False)

    if not eval_history.empty:
        eval_history.to_csv(OUTPUT_DIR / "contextual_eval_history.csv", index=False)


def plot_test_summary(summary: pd.DataFrame) -> None:
    test_df = summary[summary["split"] == "test"].copy()
    if test_df.empty:
        return

    pivot = test_df.pivot_table(
        index="experiment_name",
        columns="agent_name",
        values="mean_reward",
        aggfunc="mean",
    ).sort_index()

    ax = pivot.plot(kind="barh", figsize=(10, 6))
    ax.set_title("Test Mean Reward by Experiment")
    ax.set_xlabel("Mean Reward")
    ax.set_ylabel("Experiment")
    ax.legend(title="Agent")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_mean_reward_by_experiment.png", dpi=160)
    plt.close()


def plot_split_summary(summary: pd.DataFrame) -> None:
    pivot = summary.pivot_table(
        index=["experiment_name", "split"],
        columns="agent_name",
        values="mean_reward",
        aggfunc="mean",
    ).sort_index()

    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Mean Reward by Experiment and Split")
    ax.set_xlabel("Experiment, Split")
    ax.set_ylabel("Mean Reward")
    ax.legend(title="Agent")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mean_reward_by_experiment_and_split.png", dpi=160)
    plt.close()


def plot_eval_history(eval_history: pd.DataFrame) -> None:
    if eval_history.empty:
        return

    for agent_name, agent_df in eval_history.groupby("agent_name"):
        fig, ax = plt.subplots(figsize=(10, 5))
        for experiment_name, experiment_df in agent_df.groupby("experiment_name"):
            experiment_df = experiment_df.sort_values("eval_steps")
            ax.plot(
                experiment_df["eval_steps"],
                experiment_df["validation_mean_reward"],
                marker="o",
                label=f"{experiment_name} validation",
            )
            ax.plot(
                experiment_df["eval_steps"],
                experiment_df["train_context_mean_reward"],
                linestyle="--",
                alpha=0.6,
                label=f"{experiment_name} train",
            )

        ax.set_title(f"Evaluation History for {agent_name}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Reward")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{agent_name}_eval_history.png", dpi=160)
        plt.close(fig)


def plot_joint_context_heatmaps(per_context: pd.DataFrame) -> None:
    if per_context.empty:
        return

    joint_df = per_context[
        per_context["experiment_name"].str.contains("joint_variation", na=False)
    ].copy()
    if joint_df.empty:
        return

    for (agent_name, split), group in joint_df.groupby(["agent_name", "split"]):
        pivot = group.pivot_table(
            index="tilt_angle",
            columns="friction",
            values="mean_reward",
            aggfunc="mean",
        ).sort_index()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_title(f"{agent_name} {split} Joint Context Reward")
        ax.set_xlabel("Friction")
        ax.set_ylabel("Tilt Angle")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        plt.colorbar(im, ax=ax, label="Mean Reward")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"{agent_name}_{split}_joint_context_heatmap.png", dpi=160
        )
        plt.close(fig)


def main() -> None:
    summary, per_context, eval_history = collect_summary_frames()
    save_csv_outputs(summary, per_context, eval_history)
    plot_test_summary(summary)
    plot_split_summary(summary)
    plot_eval_history(eval_history)
    plot_joint_context_heatmaps(per_context)
    print(f"Wrote contextual analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
