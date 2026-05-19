from typing import Any, Dict, List, Optional, Tuple

import os
from collections import OrderedDict, deque
import copy
import random

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class Policy(nn.Module):
    """
    Multi-layer perceptron mapping states to action probabilities.

    Implements a linear feed-forward network with one hidden layer and softmax output.

    Parameters
    ----------
    state_space : gym.spaces.Box
        Observation space defining the dimensionality of inputs.
    action_space : gym.spaces.Discrete
        Action space defining number of output classes.
    hidden_size : int, optional
        Number of units in the hidden layer (default is 128).
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
        hidden_layer_count: int = 1,
    ):
        """
        Initialize the policy network.

        Parameters
        ----------
        state_space : gym.spaces.Box
            Observation space of the environment.
        action_space : gym.spaces.Discrete
            Action space of the environment.
        hidden_size : int, optional
            Number of hidden units. Defaults to 128.
        hidden_layer_count : int, optional
            Number of hidden layers. Defaults to 1.
        """
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.hidden_layer_count = hidden_layer_count

        if hidden_layer_count < 1:
            raise ValueError("num_linear_layers must be at least 1")

        layers: list[tuple[str, nn.Module]] = [
            ("fc1", nn.Linear(self.state_dim, hidden_size)),
            ("relu1", nn.ReLU()),
        ]

        for layer_idx in range(2, hidden_layer_count + 1):
            layers.append((f"fc{layer_idx}", nn.Linear(hidden_size, hidden_size)))
            layers.append((f"relu{layer_idx}", nn.ReLU()))

        layers.append(("out", nn.Linear(hidden_size, action_space.n)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities for given state(s).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (state_dim,) or (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over actions, shape (batch_size, n_actions).
        """
        # level 1
        # DONE: Apply fc1 followed by ReLU (Flatten input if needed)
        # fc1_out = self.relu(self.fc1(x))
        # DONE: Apply fc2 to get logits
        # logits = self.fc2(fc1_out)
        # level 2
        logits = self.net(x)
        # DONE: Return softmax over logits along the last dimension
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent(AbstractAgent):
    """
    REINFORCE agent performing on-policy Monte Carlo policy gradient updates.

    Wraps an MLP policy network and optimizer, providing train, predict, save, load, and evaluate methods.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment for interaction.
    lr : float, optional
        Learning rate for optimizer (default is 1e-2).
    gamma : float, optional
        Discount factor for returns (default is 0.99).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-2,
        gamma: float = 0.99,
        seed: int = 0,
        hidden_size: int = 128,
        hidden_layer_count: int = 1,
    ) -> None:
        """
        Initialize the REINFORCE agent.

        Args:
            env (gym.Env): Environment for training.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            seed (int, optional): Random seed. Defaults to 0.
            hidden_size (int, optional): Number of hidden units. Defaults to 128.
            hidden_layer_count (int, optional): Number of hidden layers. Defaults to 1.
        """

        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.policy = Policy(
            env.observation_space, env.action_space, hidden_size, hidden_layer_count
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.total_episodes = 0

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action according to the current policy.

        In training mode, samples stochastically and returns log probability.
        In evaluation mode, returns the argmax action deterministically.

        Parameters
        ----------
        state : np.ndarray
            Current observation from the environment.
        info : dict, optional
            Additional info (unused here, default is empty).
        evaluate : bool, optional
            If True, use deterministic policy (default is False).

        Returns
        -------
        action : int
            Selected action index.
        info_out : dict
            Contains 'log_prob' if in training mode; empty if evaluating.
        """
        # DONE: Pass state through the policy network to get action probabilities
        # If evaluate is True, return the action with highest probability
        # Otherwise, sample from the action distribution and return the log-probability as a key in the dictionary (Hint: use torch.distributions.Categorical)
        probs = self.policy(torch.from_numpy(state))
        if evaluate:
            return torch.argmax(probs).item(), {}

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        return action.item(), {"log_prob": dist.log_prob(action)}

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted reward-to-go for each timestep.

        Parameters
        ----------
        rewards : list of float
            Sequence of rewards for one episode.

        Returns
        -------
        torch.Tensor
            Discounted returns tensor of shape (len(rewards),).
        """
        # DONE: Initialize running return R = 0
        R = 0
        discounted_r = []
        # DONE: Iterate over rewards and compute the return-to-go:
        #       - Update R = r + gamma * R
        #       - Insert R at the beginning of the returns list
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_r.insert(0, R)

        # DONE: Convert the list of returns to a torch.Tensor and return
        return torch.Tensor(discounted_r)

    def update_agent(
        self,
        training_batch: List[
            Tuple[np.ndarray, int, float, np.ndarray, bool, Dict[str, Any]]
        ],
    ) -> float:
        """
        Perform a policy-gradient update using one full episode.

        Parameters
        ----------
        training_batch : list of tuples
            Each tuple is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            Scalar loss value after update.
        """
        # unpack log_probs & rewards
        log_probs = [t[5]["log_prob"] for t in training_batch]
        rewards = [t[2] for t in training_batch]

        # compute discounted returns
        returns_t = self.compute_returns(rewards)

        # normalize advantages
        # DONE: Normalize advantages with mean and standard deviation,
        # and add 1e-8 to the denominator to avoid division by zero
        advantages = (returns_t - (returns_t.mean())) / (
            returns_t.std(unbiased=False) + 1e-8
        )  # devide by n andnot (n-1)

        lp_tensor = torch.stack(log_probs)
        loss = -torch.sum(lp_tensor * advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        """
        Save policy network and optimizer state to a checkpoint.

        Parameters
        ----------
        path : str
            File path to save checkpoint.
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load policy network and optimizer state from checkpoint.

        Parameters
        ----------
        path : str
            File path of checkpoint to load.
        """
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        """
        Evaluate policy over multiple episodes.

        Parameters
        ----------
        eval_env : gym.Env
            Environment for evaluation.
        num_episodes : int, optional
            Number of episodes to run (default is 10).

        Returns
        -------
        mean_return : float
            Average episode return.
        std_return : float
            Standard deviation of returns.
        """
        self.policy.eval()
        returns: List[float] = []
        # DONE: rollout num_episodes in eval_env and aggregate undiscounted returns across episodes
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_return = 0
            steps = 0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                episode_return += reward
                steps += 1
            returns.append(episode_return)

        self.policy.train()  # Set back to training mode

        # DONE: Return the mean and std of the returns across episodes
        mean = np.mean(returns) if returns else 0.0
        std = np.std(returns) if returns else 0.0
        return mean, std

    def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> None:
        """
        Train the agent on-policy for a number of episodes.

        Parameters
        ----------
        num_episodes : int
            Total number of training episodes.
        eval_interval : int, optional
            Frequency of evaluation prints (default is 10).
        """
        eval_env = gym.make(
            self.env.spec.id, **self.env.spec.kwargs
        )  # fresh copy for eval
        best = -float("inf")
        current = 0
        train_reward_buffer = {
            "steps": [],
            "train_rewards": [],
            "episodes": [],
            "losses": [],
        }
        eval_reward_buffer = {"eval_std_ret": [], "eval_mean_ret": [], "episodes": []}
        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            batch: List[Tuple[Any, ...]] = []

            while not done:
                action, info = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                batch.append((state, action, float(reward), next_state, done, info))
                state = next_state

            loss = self.update_agent(batch)
            total_return = sum(r for _, _, r, *_ in batch)
            self.total_episodes += 1

            train_reward_buffer["steps"].append(len(batch))
            train_reward_buffer["train_rewards"].append(sum(t[2] for t in batch))
            train_reward_buffer["episodes"].append(ep)
            train_reward_buffer["losses"].append(loss)

            if ep % 10 == 0:
                print(f"[Train] Ep {ep:3d} Return {total_return:5.1f} Loss {loss:.3f}")

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate(eval_env, num_episodes=eval_episodes)
                print(f"[Eval ] Ep {ep:3d} AvgReturn {mean_ret:5.1f} ± {std_ret:4.1f}")
                eval_reward_buffer["eval_mean_ret"].append(mean_ret)
                eval_reward_buffer["eval_std_ret"].append(std_ret)
                eval_reward_buffer["episodes"].append(ep)
                current = mean_ret
                if mean_ret > best:
                    best = mean_ret
                    self.save("best_reinforce.pth")
                    print(f"New best model saved with avg return {best:.1f}")

        pd.DataFrame(train_reward_buffer).to_csv(
            os.path.abspath("train_rewards.csv"), index=False
        )
        pd.DataFrame(eval_reward_buffer).to_csv(
            os.path.abspath("eval_rewards.csv"), index=False
        )
        print("Training complete.")
        self.save("final_reinforce.pth")
        print(f"Final model avg return {current:.1f} saved.")

# Level 3 - DDPG
class ReplayBuffer:
    """Fixed-size circular replay buffer for off-policy learning.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample a random mini-batch and return stacked tensors."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    Produces mean-reverting noise suitable for continuous action spaces.

    Parameters
    ----------
    action_dim : int
        Dimensionality of the action space.
    mu : float
        Long-run mean of the process (default 0.0).
    theta : float
        Speed of mean reversion (default 0.15).
    sigma : float
        Volatility / noise scale (default 0.2).
    """

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ) -> None:
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self) -> None:
        """Reset noise state to mean."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Return next noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.action_dim
        )
        self.state = self.state + dx
        return self.state

class DDPGActor(nn.Module):
    """Deterministic policy network: state → action.

    Uses tanh output scaled to the action bounds.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation space.
    action_dim : int
        Dimensionality of the action space.
    action_high : np.ndarray
        Upper bound of each action dimension.
    hidden_size : int
        Number of units per hidden layer (default 256).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_high: np.ndarray,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.action_scale = torch.FloatTensor(action_high)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map state to a bounded deterministic action."""
        return self.net(state) * self.action_scale.to(state.device)

class DDPGCritic(nn.Module):
    """Q-network: (state, action) → scalar Q-value.

    Concatenates state and action before the first hidden layer.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation space.
    action_dim : int
        Dimensionality of the action space.
    hidden_size : int
        Number of units per hidden layer (default 256).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q(state, action)."""
        return self.net(torch.cat([state, action], dim=-1))

class DDPGAgent(AbstractAgent):
    """Deep Deterministic Policy Gradient agent for continuous action spaces.

    Implements the algorithm from Lillicrap et al. (2015):
    https://arxiv.org/pdf/1509.02971

    Key design choices that make DDPG stable (mirroring DQN insights):
    - **Replay buffer**: breaks temporal correlations between consecutive samples.
    - **Target networks**: separate, slowly-updated copies of actor and critic
      prevent the moving-target instability seen in naive Q-learning.
    - **Soft updates** (Polyak averaging) instead of hard copies keep targets
      smooth and avoid oscillations.
    - **Batch normalisation** is optional but recommended in the paper; here we
      rely on the large hidden size and Adam optimiser for stability.

    Parameters
    ----------
    env : gym.Env
        Continuous-action Gymnasium environment.
    lr_actor : float
        Learning rate for the actor network (default 1e-4).
    lr_critic : float
        Learning rate for the critic network (default 1e-3).
    gamma : float
        Discount factor (default 0.99).
    tau : float
        Soft-update coefficient for target networks (default 5e-3).
    buffer_capacity : int
        Replay buffer size (default 100_000).
    batch_size : int
        Mini-batch size for gradient updates (default 64).
    hidden_size : int
        Hidden layer width for both networks (default 256).
    noise_sigma : float
        OU noise scale for exploration (default 0.2).
    warmup_steps : int
        Number of random steps before learning starts (default 1000).
    seed : int
        Random seed (default 0).
    """

    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 5e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        hidden_size: int = 256,
        noise_sigma: float = 0.2,
        warmup_steps: int = 1000,
        seed: int = 0,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        action_high = env.action_space.high

        # Actor and its target
        self.actor = DDPGActor(state_dim, action_dim, action_high, hidden_size)
        self.actor_target = copy.deepcopy(self.actor)

        # Critic and its target
        self.critic = DDPGCritic(state_dim, action_dim, hidden_size)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimisers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer and exploration noise
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise = OUNoise(action_dim, sigma=noise_sigma)

        self.total_steps = 0

    # Core interface
    def predict_action(
        self,
        state: np.ndarray,
        info: Dict[str, Any] = {},
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select a (possibly noisy) action for the given state.

        Parameters
        ----------
        state : np.ndarray
            Current environment observation.
        info : dict, optional
            Unused; kept for API compatibility.
        evaluate : bool, optional
            If True, return the clean deterministic action (no noise).

        Returns
        -------
        action : np.ndarray
            Clipped action within the environment's action bounds.
        info_out : dict
            Empty dict (DDPG stores no per-step info).
        """
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_t).squeeze(0).numpy()
        self.actor.train()

        if not evaluate:
            action = action + self.noise.sample()

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action, {}

    def update_agent(
        self,
        training_batch: Optional[Any] = None,
    ) -> float:
        """Sample a mini-batch from the replay buffer and update actor + critic.

        The critic minimises the Bellman TD error:
            L = E[(r + γ · Q_target(s', μ_target(s')) − Q(s, a))²]

        The actor maximises the critic's Q-value estimate:
            J = E[Q(s, μ(s))]  →  gradient ascent via -mean(Q)

        Both target networks are then soft-updated:
            θ_target ← τ·θ + (1−τ)·θ_target

        Parameters
        ----------
        training_batch : ignored
            Kept for API compatibility; DDPG samples from its own buffer.

        Returns
        -------
        critic_loss : float
            Scalar TD loss for the critic update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # ---- Critic update ----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q_pred = self.critic(states, actions)
        critic_loss = F.mse_loss(q_pred, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor update ----
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft update of target networks ----
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return float(critic_loss.item())

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        """Polyak-average online network parameters into the target network.

        θ_target ← τ·θ_online + (1−τ)·θ_target
        """
        for param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # Persistence
    def save(self, path: str) -> None:
        """Save all network and optimiser states to a checkpoint file."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load all network and optimiser states from a checkpoint file."""
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    # Evaluation & training loop
    def evaluate(
        self, eval_env: gym.Env, num_episodes: int = 10
    ) -> Tuple[float, float]:
        """Evaluate the deterministic policy over several episodes.

        Parameters
        ----------
        eval_env : gym.Env
            A separate environment instance used only for evaluation.
        num_episodes : int
            Number of episodes to roll out (default 10).

        Returns
        -------
        mean_return : float
        std_return : float
        """
        returns: List[float] = []
        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                ep_return += float(reward)
            returns.append(ep_return)
        return float(np.mean(returns)), float(np.std(returns))

    def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> None:
        """Train the DDPG agent for a given number of episodes.

        During the first ``warmup_steps`` environment steps the agent takes
        uniformly random actions to pre-fill the replay buffer before any
        gradient updates are performed.

        Parameters
        ----------
        num_episodes : int
            Total number of training episodes.
        eval_interval : int
            Evaluate every this many episodes (default 10).
        eval_episodes : int
            Number of episodes per evaluation (default 5).
        """
        eval_env = gym.make(self.env.spec.id, **self.env.spec.kwargs)
        best = -float("inf")

        train_reward_buffer = {
            "steps": [],
            "train_rewards": [],
            "episodes": [],
            "losses": [],
        }
        eval_reward_buffer = {"eval_std_ret": [], "eval_mean_ret": [], "episodes": []}

        for ep in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            self.noise.reset()
            done = False
            ep_return = 0.0
            ep_loss = 0.0
            steps = 0

            while not done:
                # Warm-up: pure random exploration to fill the buffer
                if self.total_steps < self.warmup_steps:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.predict_action(state)

                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc

                self.replay_buffer.add(state, action, float(reward), next_state, done)
                state = next_state
                self.total_steps += 1
                ep_return += float(reward)

                loss = self.update_agent()
                ep_loss += loss
                steps += 1

            avg_loss = ep_loss / max(steps, 1)
            train_reward_buffer["steps"].append(steps)
            train_reward_buffer["train_rewards"].append(ep_return)
            train_reward_buffer["episodes"].append(ep)
            train_reward_buffer["losses"].append(avg_loss)

            if ep % 10 == 0:
                print(
                    f"[Train] Ep {ep:4d}  Return {ep_return:8.2f}  "
                    f"CriticLoss {avg_loss:.4f}  Steps {self.total_steps}"
                )

            if ep % eval_interval == 0:
                mean_ret, std_ret = self.evaluate(eval_env, num_episodes=eval_episodes)
                print(
                    f"[Eval ] Ep {ep:4d}  AvgReturn {mean_ret:8.2f} ± {std_ret:.2f}"
                )
                eval_reward_buffer["eval_mean_ret"].append(mean_ret)
                eval_reward_buffer["eval_std_ret"].append(std_ret)
                eval_reward_buffer["episodes"].append(ep)
                if mean_ret > best:
                    best = mean_ret
                    self.save("best_ddpg.pth")
                    print(f"  → New best saved ({best:.2f})")

        pd.DataFrame(train_reward_buffer).to_csv(
            os.path.abspath("train_rewards_ddpg.csv"), index=False
        )
        pd.DataFrame(eval_reward_buffer).to_csv(
            os.path.abspath("eval_rewards_ddpg.csv"), index=False
        )
        print("Training complete.")
        self.save("final_ddpg.pth")

@hydra.main(
    config_path="../configs/agent/", config_name="reinforce", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with fields:
          env:
            name: str        # Gym environment id
          seed: int
          agent:
            lr: float
            gamma: float
            hidden_size: int
          train:
            episodes: int
            eval_interval: int
            eval_episodes: int
    """
    # Initialize environment and seed
    print(f"config: {cfg}")

    env_kwargs = cfg.env.get("kwargs", {})
    env = gym.make(cfg.env.name, **env_kwargs)
    set_seed(env, cfg.seed)

    # Instantiate agent with hyperparameters from config
    agent = REINFORCEAgent(
        env=env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
        hidden_layer_count=cfg.agent.hidden_layer_count,
    )

    # Train agent
    agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )

@hydra.main(
    config_path="../configs/agent/", config_name="ddpg", version_base="1.1"
)
def main_ddpg(cfg: DictConfig) -> None:
    """Entry point for training the DDPG agent via Hydra.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with fields:
          env:
            name: str          # continuous-action Gym environment id
            kwargs: dict
          seed: int
          agent:
            lr_actor: float
            lr_critic: float
            gamma: float
            tau: float
            buffer_capacity: int
            batch_size: int
            hidden_size: int
            noise_sigma: float
            warmup_steps: int
          train:
            episodes: int
            eval_interval: int
            eval_episodes: int
    """
    print(f"config: {cfg}")
    env_kwargs = cfg.env.get("kwargs", {})
    env = gym.make(cfg.env.name, **env_kwargs)
    set_seed(env, cfg.seed)

    agent = DDPGAgent(
        env=env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        hidden_size=cfg.agent.hidden_size,
        noise_sigma=cfg.agent.noise_sigma,
        warmup_steps=cfg.agent.warmup_steps,
        seed=cfg.seed,
    )

    agent.train(
        num_episodes=cfg.train.episodes,
        eval_interval=cfg.train.eval_interval,
        eval_episodes=cfg.train.eval_episodes,
    )

if __name__ == "__main__":
    import sys
    if "--ddpg" in sys.argv:
        sys.argv.remove("--ddpg")
        main_ddpg()
    else:
        main()
