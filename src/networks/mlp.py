import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP to classify digits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)
