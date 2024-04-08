import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.prelu2 = nn.PReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3)
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.prelu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.in_channels != self.out_channels:
            y += self.downsample(x)
        else:
            y += x
        y = self.max_pool1(self.prelu2(y))
        return y
