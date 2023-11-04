import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        num_inp_channels: int,
        num_out_fmaps: int,
        kernel_size: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=num_inp_channels,
            out_channels=num_out_fmaps,
            kernel_size=(kernel_size, kernel_size),
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))


class CNN(nn.Module):
    """Simple CNN to classify digits."""

    def __init__(self) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d(2, 0)

        self.conv1 = ConvBlock(num_inp_channels=1, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=16, kernel_size=5)

        self.mlp = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)

        bsz, _, _, _ = x.shape
        x = x.reshape(bsz, -1)

        y = self.mlp(x)
        return y
