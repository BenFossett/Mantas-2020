from typing import Union, NamedTuple
import torch
from torch import nn
from torch.nn import functional as F

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class MantaIDNet(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)

        self.bn32 = nn.BatchNorm2d(num_features=32)
        self.bn64 = nn.BatchNorm2d(num_features=64)
        self.dropout = nn.Dropout(p=0.5)

        # Conv Layer 1 - 32 kernels with (3x3) receptive field
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv1)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 2 - 32 kernels with (5x5) receptive field
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 3 - 64 kernels with (5x5) receptive field
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv3)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 4 - 64 kernels with (3x3) receptive field
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv4)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 5 - 64 kernels with (3x3) receptive field
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv5)
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 6 - 64 kernels with (3x3) receptive field
        self.conv6 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv6)
        self.pool6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # Conv Layer 6 - 64 kernels with (3x3) receptive field
        self.conv7 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.initialise_layer(self.conv7)

        # Dense Layer 1
        self.fc1 = nn.Linear(8192, 100)
        self.initialise_layer(self.fc1)

        # Output Layer - four units
        self.out = nn.Linear(100, 100)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn32(self.conv1(images)))
        x = self.pool1(x)
        x = F.relu(self.bn32(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn64(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn64(self.conv4(x)))
        x = self.pool4(x)
        x = F.relu(self.bn64(self.conv5(self.dropout(x))))
        x = self.pool5(x)
        x = F.relu(self.bn64(self.conv6(x)))
        x = self.pool6(x)
        x = F.relu(self.conv7(self.dropout(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.out(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)
