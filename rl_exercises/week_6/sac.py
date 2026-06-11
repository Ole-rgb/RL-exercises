"""
Soft Actor-Critic (SAC) — from scratch implementation.

Based on: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor", ICML 2018.
https://arxiv.org/pdf/1801.01290

SAC key differences from PPO:
- Off-policy: learns from a replay buffer rather than fresh on-policy rollouts
  → much higher sample efficiency.
- Continuous actions: uses a reparameterised Gaussian policy (the "squashed"
  Normal via tanh) instead of a categorical distribution.
- Dual critics (Q1, Q2): takes the minimum to reduce over-estimation bias
  (clipped double-Q trick from TD3).
- Automatic entropy temperature α: a Lagrange multiplier that adapts to keep
  entropy at a target level — no manual ent_coef tuning required.
- Soft updates (Polyak averaging) for the target critics.
"""

import copy
import os
import random
from collections import deque
from typing import Any, Deque, List, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent

# Seeding helpers
def set_seed(env: gym.Env, seed: int = 0) -> None:
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Network definitions
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6  # for log-sum numerical stability when computing entropy

class SACGaussianPolicy(nn.Module):
    """
    Squashed-Gaussian stochastic policy for continuous action spaces.

    Outputs (mean, log_std) and supports:
      - sample(): reparameterised sample + log-prob (with tanh correction)
      - evaluate(): deterministic tanh(mean) for greedy evaluation
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reparameterised sample with log-prob corrected for tanh squashing.

        Returns
        -------
        action   : scaled action in [action_low, action_high]
        log_prob : log π(a|s), shape (batch,)
        mean     : deterministic tanh(mean) (for optional use)
        """
        mean, log_std = self.forward(x)
        std = log_std.exp()
        # Reparameterisation trick: z = mean + std * ε, ε ~ N(0,1)
        eps = torch.randn_like(mean)
        z = mean + std * eps  # pre-squash

        # Squash to (-1, 1) then scale to actual action range
        tanh_z = torch.tanh(z)
        action = tanh_z * self.action_scale + self.action_bias

        # Log-prob with tanh Jacobian correction:
        #   log π(a|s) = log N(z; μ, σ) - Σ log(1 - tanh(z)²)
        log_prob = (
            torch.distributions.Normal(mean, std).log_prob(z)
            - torch.log(self.action_scale * (1.0 - tanh_z.pow(2)) + EPSILON)
        ).sum(dim=-1)

        det_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, det_action

class SACCritic(nn.Module):
    """
    Twin Q-networks Q1(s,a), Q2(s,a) — the clipped double-Q trick.
    Using two separate networks (not just two heads) as in the original paper.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        def make_q():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

# Replay buffer
class ReplayBuffer:
    """Simple circular replay buffer for SAC."""

    def __init__(self, capacity: int = 1_000_000):
        self.buf: Deque[Any] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buf.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buf)

# SAC Agent
class SACAgent(AbstractAgent):
    """
    Full SAC implementation with automatic entropy tuning.

    Key paper claims this implementation tests:
      1. Better sample efficiency than on-policy methods (PPO) due to the
         replay buffer — each transition is reused many times.
      2. Robust to hyperparameter choice: automatic α means you don't need to
         hand-tune the entropy coefficient.
      3. Stable training: dual critics prevent Q over-estimation; soft Polyak
         targets prevent oscillation.
      4. Competitive final performance on continuous-control benchmarks.
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,          # Polyak averaging coefficient
        alpha: float = 0.2,          # initial entropy temperature
        auto_alpha: bool = True,     # automatic entropy tuning
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        learning_starts: int = 1000, # steps before first gradient update
        update_freq: int = 1,        # gradient updates per env step
        hidden_size: int = 256,
        seed: int = 0,
    ) -> None:
        set_seed(env, seed)
        self.seed = seed
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.update_freq = update_freq

        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        # Scale/bias to map tanh ∈ (-1,1) → actual action range
        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Networks
        self.actor = SACGaussianPolicy(
            obs_dim, action_dim, self.action_scale, self.action_bias, hidden_size
        )
        self.critic = SACCritic(obs_dim, action_dim, hidden_size)
        # Target critic (soft-updated copy — never trained directly)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning (Appendix of SAC paper)
        # Target entropy = -|A| (heuristic from the paper)
        self.auto_alpha = auto_alpha
        self.target_entropy = -float(action_dim)
        if auto_alpha:
            # log α is the optimisation variable (unconstrained in log-space)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay = ReplayBuffer(buffer_size)

    # AbstractAgent interface
    def predict_action(  # type: ignore[override]
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        """Sample (or greedily select) an action."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob, det_action = self.actor.sample(s)
            if evaluate:
                return det_action.squeeze(0).numpy(), None
            return action.squeeze(0).numpy(), log_prob

    def save(self, path: str = "sac_checkpoint.pt") -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str = "sac_checkpoint.pt") -> None:
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

    # Core update step
    def update_agent(  # type: ignore[override]
        self, *args: Any, **kwargs: Any
    ) -> Tuple[float, float, float, float]:
        """Single SAC gradient step from the replay buffer."""
        if len(self.replay) < self.batch_size:
            return 0.0, 0.0, 0.0, self.alpha

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        # Critic update
        with torch.no_grad():
            # Next actions + log-probs from current policy
            next_a, next_logp, _ = self.actor.sample(next_states)
            # Soft Bellman target: r + γ·(min Q_target - α·log π)
            q1_next, q2_next = self.critic_target(next_states, next_a)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + self.gamma * (1.0 - dones) * (
                min_q_next - self.alpha * next_logp
            )

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        new_a, new_logp, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_a)
        min_q_new = torch.min(q1_new, q2_new)
        # Maximise E[Q - α·log π] ↔ minimise -(Q - α·log π)
        actor_loss = (self.alpha * new_logp - min_q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Alpha (entropy temperature) update
        alpha_loss = 0.0
        if self.auto_alpha:
            # Gradient of dual objective w.r.t. log α:
            #   L(α) = E[-α·(log π + target_H)]
            alpha_loss_t = -(
                self.log_alpha * (new_logp + self.target_entropy).detach()
            ).mean()
            self.alpha_optim.zero_grad()
            alpha_loss_t.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss_t.item()

        # Polyak (soft) target update
        # θ_target ← τ·θ + (1-τ)·θ_target
        for p, p_tgt in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return (
            actor_loss.item(),
            critic_loss.item(),
            alpha_loss,
            self.alpha,
        )

    # Evaluation
    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        returns: List[float] = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            total_r = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, r, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_r += r
            returns.append(total_r)
        return float(np.mean(returns)), float(np.std(returns))

    # ── Training loop ────────────────────────────────────────────────────────

    def train(
        self,
        total_steps: int,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        step_count = 0
        episode = 0

        while step_count < total_steps:
            state, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0

            while not done and step_count < total_steps:
                # Random exploration before learning starts
                if step_count < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.predict_action(state)

                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                # Store "done" only on termination (not truncation) so the
                # bootstrap target is correct at episode boundaries.
                self.replay.add(state, action, reward, next_state, float(term))
                state = next_state
                ep_reward += reward
                ep_steps += 1
                step_count += 1

                # Gradient updates
                if step_count >= self.learning_starts and step_count % self.update_freq == 0:
                    for _ in range(self.update_freq):
                        self.update_agent()

                if step_count % eval_interval == 0:
                    mean_r, std_r = self.evaluate(
                        eval_env, num_episodes=eval_episodes
                    )
                    print(
                        f"[Eval ] Step {step_count:6d} AvgReturn {mean_r:7.1f} ± {std_r:5.1f}  α={self.alpha:.4f}"
                    )

            episode += 1
            print(
                f"[Train] Episode {episode:4d} Step {step_count:6d} Return {ep_reward:7.1f}"
            )

        print("Training complete.")

# Hydra entry-point
@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    agent = SACAgent(
        env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        alpha=cfg.agent.alpha,
        auto_alpha=cfg.agent.auto_alpha,
        batch_size=cfg.agent.batch_size,
        buffer_size=cfg.agent.buffer_size,
        learning_starts=cfg.agent.learning_starts,
        update_freq=cfg.agent.update_freq,
        hidden_size=cfg.agent.hidden_size,
        seed=cfg.seed,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
