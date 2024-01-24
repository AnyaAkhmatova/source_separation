import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.prelu2 = nn.PReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3)

    def forward(self, x):
        y = self.prelu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += x
        y = self.max_pool1(self.prelu2(y))
        return y
