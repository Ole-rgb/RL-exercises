# Week 4 Level 3 Runner

This directory contains the level 3 DQN extensions:

- `extended_dqn.py`: extended DQN entrypoint with configurable replay buffer and DQN mode
- `per.py`: proportional prioritized experience replay
- `run_extended_dqn_sweep.py`: small launcher for running parameter sweeps over `extended_dqn`

## Basic usage

Run the sweep launcher with its default grid:

```bash
python -m rl_exercises.week_4.level_3.run_extended_dqn_sweep
```

The runner launches:

```bash
python -m rl_exercises.week_4.level_3.extended_dqn <hydra overrides...>
```

Every dimension can either:

- sweep over multiple values, or
- be fixed by passing exactly one value

Available sweep dimensions:

- `--seeds`
- `--modes`
- `--buffer-types`
- `--batch-sizes`
- `--buffer-capacities`
- `--hidden-dims`
- `--num-linear-layers`

You can also pass extra Hydra overrides through:

```bash
--extra-overrides train.num_frames=20000 train.eval_interval=1000 outpath=results/dqn
```

## Experiment 1

Sweep:

- `batch_size`
- `buffer.capacity`
- `network.hidden_dim`
- `network.num_linear_layers`

Do not sweep:

- `agent.mode`
- `buffer.type`

Example:

```bash
python -m rl_exercises.week_4.level_3.run_extended_dqn_sweep \
  --modes normal \
  --buffer-types uniform \
  --batch-sizes 32,64 \
  --buffer-capacities 100,1000,10000 \
  --hidden-dims 32,64,128 \
  --num-linear-layers 2,3,4 \
  --extra-overrides train.num_frames=20000 train.eval_interval=1000 outpath=results/extended_dqn/exp1
```

This runs plain DQN through `extended_dqn`, with a uniform replay buffer, while sweeping the architecture and replay-capacity settings.

## Experiment 2

Sweep only:

- `agent.mode`
- `buffer.type`

Do not sweep:

- `batch_size`
- `buffer.capacity`
- `network.hidden_dim`
- `network.num_linear_layers`

Example:

```bash
python -m rl_exercises.week_4.level_3.run_extended_dqn_sweep \
  --modes normal double_dqn \
  --buffer-types uniform prioritized \
  --batch-sizes 32 \
  --buffer-capacities 10000 \
  --hidden-dims 64 \
  --num-linear-layers 2 \
  --extra-overrides train.num_frames=20000 train.eval_interval=1000 outpath=results/extended_dqn/exp2
```

This gives the four core comparisons:

- normal + uniform
- normal + prioritized
- double_dqn + uniform
- double_dqn + prioritized
