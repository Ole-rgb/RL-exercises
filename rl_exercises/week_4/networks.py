from collections import OrderedDict

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple MLP mapping state → Q‐values for each action.

    Architecture:
      Input → hidden layers (Linear + ReLU) → Linear(hidden_dim→n_actions)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        num_linear_layers: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        obs_dim : int
            Dimensionality of observation space.
        n_actions : int
            Number of discrete actions.
        hidden_dim : int
            Hidden layer size.
        num_linear_layers : int
            Number of hidden linear layers before the output layer.
        """
        super().__init__()
        if num_linear_layers < 1:
            raise ValueError("num_linear_layers must be at least 1")

        layers: list[tuple[str, nn.Module]] = [
            ("fc1", nn.Linear(obs_dim, hidden_dim)),
            ("relu1", nn.ReLU()),
        ]

        for layer_idx in range(2, num_linear_layers + 1):
            layers.append((f"fc{layer_idx}", nn.Linear(hidden_dim, hidden_dim)))
            layers.append((f"relu{layer_idx}", nn.ReLU()))

        layers.append(("out", nn.Linear(hidden_dim, n_actions)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of states, shape (batch, obs_dim).

        Returns
        -------
        torch.Tensor
            Q‐values, shape (batch, n_actions).
        """
        return self.net(x)
