from __future__ import annotations

from pathlib import Path

import pandas as pd
from rl_exercises.plotting.rliable_results import (
    PlotFilters,
    apply_filters,
    collect_results,
    extract_curve_scores,
    parse_result_metadata,
    parse_seed_folder,
    plot_results,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_parse_seed_folder_handles_plain_and_hyperparameter_runs():
    assert parse_seed_folder("seed_0") == (0, "default")
    assert parse_seed_folder("seed_42.gamma_0.95.alpha_0.1") == (
        42,
        "seed_42.gamma_0.95.alpha_0.1",
    )
    assert parse_seed_folder("not_a_seed") is None


def test_parse_result_metadata_contextual_legacy_and_hyperparameter_layouts(
    tmp_path: Path,
):
    results_dir = tmp_path / "results"

    contextual = (
        results_dir
        / "contextual_value_iteration"
        / "TiltedMarsRover"
        / "joint_variation"
        / "seed_0"
        / "eval_rewards.csv"
    )
    legacy = results_dir / "qlearning" / "MarsRover" / "seed_1" / "eval_rewards.csv"
    hyper = (
        results_dir
        / "tdlambda"
        / "MarsRover"
        / "seed_2.gamma_0.95.alpha_0.1"
        / "eval_rewards.csv"
    )

    contextual_meta = parse_result_metadata(contextual, results_dir)
    legacy_meta = parse_result_metadata(legacy, results_dir)
    hyper_meta = parse_result_metadata(hyper, results_dir)

    assert contextual_meta is not None
    assert contextual_meta.agent_name == "contextual_value_iteration"
    assert contextual_meta.env_name == "TiltedMarsRover"
    assert contextual_meta.experiment_name == "joint_variation"
    assert contextual_meta.seed == 0
    assert contextual_meta.run_variant == "default"

    assert legacy_meta is not None
    assert legacy_meta.agent_name == "qlearning"
    assert legacy_meta.env_name == "MarsRover"
    assert legacy_meta.experiment_name == "default"
    assert legacy_meta.seed == 1

    assert hyper_meta is not None
    assert hyper_meta.experiment_name == "default"
    assert hyper_meta.seed == 2
    assert hyper_meta.run_variant == "seed_2.gamma_0.95.alpha_0.1"


def test_extract_curve_scores_supports_contextual_eval_columns(tmp_path: Path):
    results_dir = tmp_path / "results"
    path = (
        results_dir
        / "contextual_value_iteration"
        / "TiltedMarsRover"
        / "joint_variation"
        / "seed_0"
        / "eval_rewards.csv"
    )
    _write_csv(
        path,
        [
            {
                "eval_steps": 0,
                "train_context_mean_reward": 1.0,
                "validation_mean_reward": 2.0,
            },
            {
                "eval_steps": 10,
                "train_context_mean_reward": 3.0,
                "validation_mean_reward": 4.0,
            },
        ],
    )
    metadata = parse_result_metadata(path, results_dir)
    assert metadata is not None

    extracted = extract_curve_scores(path, metadata)

    assert set(extracted["score_name"]) == {
        "train_context_mean_reward",
        "validation_mean_reward",
    }
    assert set(extracted["step"]) == {0, 10}
    assert set(extracted["agent_name"]) == {"contextual_value_iteration"}


def test_collect_results_and_filters(tmp_path: Path):
    results_dir = tmp_path / "results"
    _write_csv(
        results_dir / "qlearning" / "MarsRover" / "seed_0" / "eval_rewards.csv",
        [
            {"eval_steps": 0, "eval_rewards": 1.0},
            {"eval_steps": 10, "eval_rewards": 2.0},
        ],
    )
    _write_csv(
        results_dir / "sarsa" / "MarsRover" / "seed_0" / "eval_rewards.csv",
        [
            {"eval_steps": 0, "eval_rewards": 3.0},
            {"eval_steps": 10, "eval_rewards": 4.0},
        ],
    )

    curves, finals = collect_results(results_dir)
    filtered_curves = apply_filters(
        curves,
        PlotFilters(agent="qlearning", env="MarsRover", score="eval_rewards"),
    )
    filtered_finals = apply_filters(finals, PlotFilters(agent="qlearning"))

    assert set(curves["agent_name"]) == {"qlearning", "sarsa"}
    assert set(filtered_curves["agent_name"]) == {"qlearning"}
    assert set(filtered_curves["score_name"]) == {"eval_rewards"}
    assert set(filtered_finals["agent_name"]) == {"qlearning"}
    assert set(filtered_finals["score_name"]) == {"final_eval_rewards"}


def test_plot_results_writes_tidy_csvs_and_plots(tmp_path: Path):
    results_dir = tmp_path / "results"
    for agent, base_reward in (("qlearning", 1.0), ("sarsa", 2.0)):
        for seed in (0, 1):
            _write_csv(
                results_dir / agent / "MarsRover" / f"seed_{seed}" / "eval_rewards.csv",
                [
                    {"eval_steps": 0, "eval_rewards": base_reward + seed},
                    {"eval_steps": 10, "eval_rewards": base_reward + seed + 1},
                ],
            )

    output_dir = tmp_path / "plots"
    summary = plot_results(
        results_dir=results_dir,
        output_dir=output_dir,
        filters=PlotFilters(env="MarsRover"),
        reps=5,
    )

    assert summary.tidy_curve_csv is not None
    assert summary.tidy_curve_csv.exists()
    assert summary.tidy_final_csv is not None
    assert summary.tidy_final_csv.exists()
    assert summary.plot_paths
    assert all(path.exists() for path in summary.plot_paths)
