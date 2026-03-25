import copy

import torch
import torch.nn as nn


class DoubleDQNNetwork(nn.Module):
    def __init__(self, input_channels: int, output_dim: int) -> None:
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor, branch: str = "online") -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.float() / 255.0
        network = self.online if branch == "online" else self.target
        return network(inputs)
