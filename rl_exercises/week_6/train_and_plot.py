"""
Week 6 — Training & Plotting Script

Runs experiments for all three levels and produces RLiable plots.

Usage
-----
# Level 1 – Actor-Critic baselines on CartPole and LunarLander
python train_and_plot.py --level 1

# Level 2 – PPO variants on LunarLander
python train_and_plot.py --level 2

# Level 3 – SAC on LunarLanderContinuous
python train_and_plot.py --level 3

# All levels at once
python train_and_plot.py --level all

Results are saved as PNG files in the current working directory:
  week6_l1_cartpole.png
  week6_l1_lunarlander.png
  week6_l2_lunarlander.png
  week6_l3_sac_vs_ppo.png
"""

import argparse
import warnings
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

# RLiable
try:
    from rliable import library as rly
    from rliable import metrics, plot_utils

    HAS_RLIABLE = True
except ImportError:
    warnings.warn("rliable not installed — falling back to plain matplotlib plots.")
    HAS_RLIABLE = False

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt

from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed as ac_set_seed
from rl_exercises.week_6.ppo import PPOAgent, set_seed as ppo_set_seed
from rl_exercises.week_6.sac import SACAgent, set_seed as sac_set_seed

# Generic train helper
EvalPoint = Tuple[int, float]  # (step, mean_return)

def train_actor_critic(
    env_name: str,
    baseline_type: str,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
    **kwargs,
) -> List[EvalPoint]:
    """Train one ActorCritic run and return (step, mean_return) pairs."""
    env = gym.make(env_name)
    ac_set_seed(env, seed)
    agent = ActorCriticAgent(
        env,
        baseline_type=baseline_type,
        seed=seed,
        **kwargs,
    )
    eval_env = gym.make(env_name)
    step_count = 0
    results: List[EvalPoint] = []

    while step_count < total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done and step_count < total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append((state, action, float(reward), next_state, done, logp))
            state = next_state
            step_count += 1

            if step_count % eval_interval == 0:
                mean_r, _ = agent.evaluate(eval_env, num_episodes=eval_episodes)
                results.append((step_count, mean_r))
                print(
                    f"  [AC/{baseline_type}/seed={seed}] step={step_count} mean_r={mean_r:.1f}"
                )

        if trajectory:
            agent.update_agent(trajectory)

    env.close()
    eval_env.close()
    return results

def train_ppo(
    env_name: str,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
    **kwargs,
) -> List[EvalPoint]:
    """Train one PPO run."""
    env = gym.make(env_name)
    ppo_set_seed(env, seed)
    agent = PPOAgent(env, seed=seed, total_steps=total_steps, **kwargs)
    eval_env = gym.make(env_name)
    step_count = 0
    results: List[EvalPoint] = []

    while step_count < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        trajectory = []

        while not done and step_count < total_steps:
            action, logp, ent, val = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )
            state = next_state
            step_count += 1

            if step_count % eval_interval == 0:
                mean_r, _ = agent.evaluate(eval_env, num_episodes=eval_episodes)
                results.append((step_count, mean_r))
                print(
                    f"  [PPO/seed={seed}] step={step_count} mean_r={mean_r:.1f}"
                )

        if trajectory:
            agent.update(trajectory, step_count)

    env.close()
    eval_env.close()
    return results

def train_sac(
    env_name: str,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    seed: int,
    **kwargs,
) -> List[EvalPoint]:
    """Train one SAC run."""
    env = gym.make(env_name)
    sac_set_seed(env, seed)
    agent = SACAgent(env, seed=seed, **kwargs)
    eval_env = gym.make(env_name)
    step_count = 0
    results: List[EvalPoint] = []

    while step_count < total_steps:
        state, _ = env.reset()
        done = False

        while not done and step_count < total_steps:
            if step_count < agent.learning_starts:
                action = env.action_space.sample()
            else:
                action, _ = agent.predict_action(state)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.replay.add(state, action, reward, next_state, float(term))
            state = next_state
            step_count += 1

            if step_count >= agent.learning_starts and step_count % agent.update_freq == 0:
                agent.update_agent()

            if step_count % eval_interval == 0:
                mean_r, _ = agent.evaluate(eval_env, num_episodes=eval_episodes)
                results.append((step_count, mean_r))
                print(
                    f"  [SAC/seed={seed}] step={step_count} mean_r={mean_r:.1f}"
                )

    env.close()
    eval_env.close()
    return results

# Plotting helpers
def build_score_matrix(
    runs: List[List[EvalPoint]],
) -> np.ndarray:
    """
    Build an (algorithms=1, runs, eval_points) matrix for RLiable.
    Each run must have the same number of eval points.
    """
    n_seeds = len(runs)
    n_points = min(len(r) for r in runs)
    mat = np.array([[pt[1] for pt in run[:n_points]] for run in runs])
    # shape: (n_seeds, n_points) → rliable expects (n_seeds, n_tasks)
    return mat  # we treat each eval point as a "task"


def plot_rliable(
    algo_results: Dict[str, List[List[EvalPoint]]],
    title: str,
    save_path: str,
) -> None:
    """
    Plot IQM (or mean) ± 95% CI using RLiable, or fall back to matplotlib.
    """
    # Collect eval step indices (use the first algo/seed as reference)
    ref_runs = next(iter(algo_results.values()))
    n_points = min(len(r) for r in ref_runs)
    steps = [ref_runs[0][i][0] for i in range(n_points)]

    if HAS_RLIABLE:
        # Build score dict: {algo: (n_seeds, n_eval_points)}
        score_dict: Dict[str, np.ndarray] = {}
        for algo, runs in algo_results.items():
            mat = np.array([[pt[1] for pt in run[:n_points]] for run in runs])
            score_dict[algo] = mat  # (n_seeds, n_eval_points)

        # For each eval step, compute IQM and 95% bootstrap CI across seeds.
        # rly.get_interval_estimates expects shape (n_seeds, n_tasks).
        # We call it once per time-step with a (n_seeds, 1) slice.
        iqm_means: Dict[str, np.ndarray] = {k: np.zeros(n_points) for k in score_dict}
        iqm_lo:    Dict[str, np.ndarray] = {k: np.zeros(n_points) for k in score_dict}
        iqm_hi:    Dict[str, np.ndarray] = {k: np.zeros(n_points) for k in score_dict}

        for i in range(n_points):
            slice_dict = {k: v[:, i : i + 1] for k, v in score_dict.items()}
            point_ests, cis = rly.get_interval_estimates(
                slice_dict, metrics.aggregate_iqm, reps=200
            )
            for k in score_dict:
                iqm_means[k][i] = point_ests[k]          # scalar
                iqm_lo[k][i]    = cis[k][0]              # lower bound
                iqm_hi[k][i]    = cis[k][1]              # upper bound

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10.colors
        for idx, algo in enumerate(score_dict):
            c = colors[idx % len(colors)]
            ax.plot(steps, iqm_means[algo], label=algo, color=c, lw=2)
            ax.fill_between(steps, iqm_lo[algo], iqm_hi[algo], alpha=0.2, color=c)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("IQM Return")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ RLiable plot saved → {save_path}")

    else:
        # Fallback: plain mean ± std
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.cm.tab10.colors
        for idx, (algo, runs) in enumerate(algo_results.items()):
            c = colors[idx % len(colors)]
            mat = np.array([[pt[1] for pt in run[:n_points]] for run in runs])
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            ax.plot(steps, mean, label=algo, color=c, lw=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=c)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Return")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  ✓ Fallback plot saved → {save_path}")

# Level runners
SEEDS = [0, 1, 2]  # 3 seeds for IQM confidence intervals
def run_level1() -> None:
    print("\n" + "=" * 60)
    print("LEVEL 1 — Actor-Critic baselines")
    print("=" * 60)

    baselines = ["none", "avg", "value", "gae"]
    common = dict(total_steps=200_000, eval_interval=10_000, eval_episodes=5)

    for env_name, tag in [("CartPole-v1", "cartpole"), ("LunarLander-v3", "lunarlander")]:
        print(f"\n  Environment: {env_name}")
        algo_results: Dict[str, List[List[EvalPoint]]] = {}
        for bl in baselines:
            runs = []
            for seed in SEEDS:
                print(f"    baseline={bl}, seed={seed}")
                run = train_actor_critic(env_name, bl, seed=seed, **common)
                runs.append(run)
            algo_results[f"AC-{bl}"] = runs

        plot_rliable(
            algo_results,
            title=f"Level 1 – Actor-Critic Baselines on {env_name}",
            save_path=f"week6_l1_{tag}.png",
        )

def run_level2() -> None:
    print("\n" + "=" * 60)
    print("LEVEL 2 — PPO variants on LunarLander-v3")
    print("=" * 60)

    env_name = "LunarLander-v3"
    common = dict(
        total_steps=200_000,
        eval_interval=10_000,
        eval_episodes=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=4,
        batch_size=64,
        ent_coef=0.01,
        vf_coef=0.5,
        hidden_size=128,
    )

    algo_results: Dict[str, List[List[EvalPoint]]] = {}

    # Actor-Critic GAE baseline (best from L1) for comparison
    print("\n  Actor-Critic (GAE, for comparison)...")
    ac_runs = []
    for seed in SEEDS:
        run = train_actor_critic(env_name, "gae", seed=seed,
                                  total_steps=200_000, eval_interval=10_000,
                                  eval_episodes=5)
        ac_runs.append(run)
    algo_results["AC-GAE"] = ac_runs

    # PPO vanilla (no LR annealing, no KL stopping)
    print("\n  PPO vanilla...")
    ppo_vanilla_runs = []
    for seed in SEEDS:
        run = train_ppo(env_name, seed=seed, lr_anneal=False, use_kl_stopping=False, **common)
        ppo_vanilla_runs.append(run)
    algo_results["PPO-vanilla"] = ppo_vanilla_runs

    # PPO enhanced (LR annealing + KL early stopping)
    print("\n  PPO enhanced (LR annealing + KL stopping)...")
    ppo_enh_runs = []
    for seed in SEEDS:
        run = train_ppo(env_name, seed=seed, lr_anneal=True, use_kl_stopping=True,
                        kl_target=0.01, **common)
        ppo_enh_runs.append(run)
    algo_results["PPO-enhanced"] = ppo_enh_runs

    plot_rliable(
        algo_results,
        title="Level 2 – PPO Variants on LunarLander-v3",
        save_path="week6_l2_lunarlander.png",
    )

def run_level3() -> None:
    print("\n" + "=" * 60)
    print("LEVEL 3 — SAC vs PPO (Continuous Control)")
    print("=" * 60)

    # SAC: continuous LunarLander
    sac_env = "LunarLanderContinuous-v3"
    # PPO: discrete LunarLander (for reference — different action space)
    # For a fair continuous-control comparison we run both on the continuous env.
    # PPO can handle continuous actions via a Gaussian policy, but our
    # implementation uses Categorical, so we compare on LunarLanderContinuous
    # with SAC and note the PPO-GAE numbers from L2 as reference.

    algo_results: Dict[str, List[List[EvalPoint]]] = {}

    print(f"\n  SAC on {sac_env}...")
    sac_runs = []
    for seed in SEEDS:
        run = train_sac(
            sac_env,
            total_steps=200_000,
            eval_interval=10_000,
            eval_episodes=5,
            seed=seed,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            auto_alpha=True,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=1000,
            update_freq=1,
            hidden_size=256,
        )
        sac_runs.append(run)
    algo_results["SAC"] = sac_runs

    # Re-run PPO-enhanced on the *discrete* LunarLander for comparison
    # (included so the plot shows both curves; note different env variants)
    ppo_env = "LunarLander-v3"
    print(f"\n  PPO-enhanced on {ppo_env} (for comparison)...")
    ppo_runs = []
    for seed in SEEDS:
        run = train_ppo(
            ppo_env,
            total_steps=200_000,
            eval_interval=10_000,
            eval_episodes=5,
            seed=seed,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            epochs=4,
            batch_size=64,
            ent_coef=0.01,
            vf_coef=0.5,
            hidden_size=128,
            lr_anneal=True,
            use_kl_stopping=True,
            kl_target=0.01,
        )
        ppo_runs.append(run)
    algo_results["PPO-enhanced (discrete)"] = ppo_runs

    plot_rliable(
        algo_results,
        title="Level 3 – SAC (Continuous) vs PPO (Discrete) LunarLander",
        save_path="week6_l3_sac_vs_ppo.png",
    )

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 6 training + plotting")
    parser.add_argument(
        "--level",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which level to run (default: all)",
    )
    args = parser.parse_args()

    if args.level in ("1", "all"):
        run_level1()
    if args.level in ("2", "all"):
        run_level2()
    if args.level in ("3", "all"):
        run_level3()
