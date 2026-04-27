# Week 2 Contextual Planners

This README covers the dedicated contextual planning entrypoint in `rl_exercises/week_2/contextual_train_agent.py`.

## Supported Agents

- `ContextualPolicyIteration`
- `ContextualValueIteration`
- `PolicyIteration` as a non-contextual baseline
- `ValueIteration` as a non-contextual baseline

These planners expect the contextual rover environment:

- `env_name: TiltedMarsRover`
- context features: `tilt_angle` and `friction`

## Run Commands

```bash
# Contextual Policy Iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_policy_iteration

# Contextual Value Iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_value_iteration
```

## Experiment Presets

```bash
# Only tilt varies during training with value iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_tilt_only

# Only friction varies during training with value iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_friction_only

# Tilt and friction vary jointly during training with value iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_joint_variation
```

Policy-iteration presets are available as separate exercises:

```bash
# Only tilt varies during training with policy iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_policy_tilt_only

# Only friction varies during training with policy iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_policy_friction_only

# Tilt and friction vary jointly during training with policy iteration
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_contextual_policy_joint_variation
```

Non-contextual baselines are available with the same three scenarios:

```bash
# Non-contextual value iteration baselines
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_value_tilt_only
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_value_friction_only
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_value_joint_variation

# Non-contextual policy iteration baselines
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_policy_tilt_only
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_policy_friction_only
python rl_exercises/week_2/contextual_train_agent.py +exercise=w2_non_contextual_policy_joint_variation
```

To run all six contextual experiments in one go with separate log files:

```bash
bash rl_exercises/week_2/run_all_contextual_experiments.sh
```

Logs are written to `results/contextual_logs/<exercise>.log`.

To summarize all contextual runs and generate plots:

```bash
python rl_exercises/week_2/summarize_contextual_results.py
```

Analysis outputs are written to `results/contextual_analysis/`.

## Config Files

- `rl_exercises/configs/exercise/w2_contextual_policy_iteration.yaml`
- `rl_exercises/configs/exercise/w2_contextual_value_iteration.yaml`
- `rl_exercises/configs/exercise/w2_contextual_tilt_only.yaml`
- `rl_exercises/configs/exercise/w2_contextual_friction_only.yaml`
- `rl_exercises/configs/exercise/w2_contextual_joint_variation.yaml`
- `rl_exercises/configs/exercise/w2_contextual_policy_tilt_only.yaml`
- `rl_exercises/configs/exercise/w2_contextual_policy_friction_only.yaml`
- `rl_exercises/configs/exercise/w2_contextual_policy_joint_variation.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_value_tilt_only.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_value_friction_only.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_value_joint_variation.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_policy_tilt_only.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_policy_friction_only.yaml`
- `rl_exercises/configs/exercise/w2_non_contextual_policy_joint_variation.yaml`
- `rl_exercises/configs/agent/contextual_policy_iteration.yaml`
- `rl_exercises/configs/agent/contextual_value_iteration.yaml`
- `rl_exercises/configs/agent/non_contextual_policy_iteration.yaml`
- `rl_exercises/configs/agent/non_contextual_value_iteration.yaml`

## What The Runner Does

`contextual_train_agent.py` keeps the contextual workflow separate from the root trainer. It:

- builds `TiltedMarsRover`
- instantiates a contextual planner
- rotates or samples training contexts across episodes
- evaluates using the context exposed by the environment in `info["context"]`
- reports train and validation performance during training
- evaluates test performance only at the end

## Notes

- The root `rl_exercises/train_agent.py` is left for the non-contextual training flow.
- The training schedule is controlled with `context_schedule`, currently `round_robin` or `random`.
- The runner writes `train_rewards.csv`, `eval_rewards.csv`, `final_eval_summary.csv`, and `final_eval_per_context.csv`.
- Each experiment scenario gets its own results subdirectory under `results/<agent>/<env>/<experiment_name>/seed_<seed>`.
