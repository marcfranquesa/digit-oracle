import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        num_inp_channels: int,
        num_out_fmaps: int,
        kernel_size: int,
        pool_size: int = 2,
        padding: int = 1,
        convolutions: int = 1,
    ) -> None:
        super().__init__()
        self.convolutions = convolutions
        self.conv = [
            nn.Conv2d(
                in_channels=num_inp_channels,
                out_channels=num_out_fmaps,
                kernel_size=(kernel_size, kernel_size),
                padding=padding,
            )
        ]
        for _ in range(convolutions - 1):
            self.conv.append(
                nn.Conv2d(
                    in_channels=num_out_fmaps,
                    out_channels=num_out_fmaps,
                    kernel_size=(kernel_size, kernel_size),
                    padding=padding,
                )
            )

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for convolution in self.conv:
            x = self.relu(convolution(x))
        return self.maxpool(x)


class VGG16(nn.Module):
    """VGG16 adapted to classify digits."""

    def __init__(self) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d(2, 0)

        self.conv1 = ConvBlock(
            num_inp_channels=1,
            num_out_fmaps=64,
            kernel_size=3,
            padding=1,
            convolutions=2,
        )
        self.conv2 = ConvBlock(
            num_inp_channels=64,
            num_out_fmaps=128,
            kernel_size=3,
            padding=1,
            convolutions=2,
        )
        self.conv3 = ConvBlock(
            num_inp_channels=128,
            num_out_fmaps=256,
            kernel_size=3,
            padding=1,
            convolutions=3,
        )
        self.conv4 = ConvBlock(
            num_inp_channels=256,
            num_out_fmaps=512,
            kernel_size=3,
            padding=1,
            convolutions=3,
        )
        self.conv5 = ConvBlock(
            num_inp_channels=512,
            num_out_fmaps=512,
            kernel_size=3,
            padding=1,
            convolutions=3,
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        bsz, _, _, _ = x.shape
        x = x.reshape(bsz, -1)

        y = self.mlp(x)
        return y
