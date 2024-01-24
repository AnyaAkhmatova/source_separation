import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, n_channels, eps=1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(1, n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))            

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = (((x - mean)**2).mean(dim=(1, 2), keepdim=True) + self.eps)**0.5
        y = (x - mean) / std
        y = self.weight * y + self.bias
        return y


class TCNBlock(nn.Module):
    def __init__(self, n_channels, dilation):
        super().__init__()
        self.n_channels = n_channels
        self.dilation = dilation

        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(n_channels)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(n_channels)
        self.conv3 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

    def forward(self, x):
        y = self.gln1(self.prelu1(self.conv1(x)))
        y = nn.functional.pad(y, (0, self.dilation * 2)) 
        y = self.gln2(self.prelu2(self.conv2(y)))
        y = self.conv3(y) + x
        return y


class TCNBlockRef(nn.Module):
    def __init__(self, n_channels, dilation, ref_dim):
        super().__init__()
        self.n_channels = n_channels
        self.dilation = dilation
        self.ref_dim = ref_dim

        self.conv1 = nn.Conv1d(n_channels + ref_dim, n_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(n_channels)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=3, dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(n_channels)
        self.conv3 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

    def forward(self, x, ref):
        ref = ref.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = torch.cat([x, ref], 1)
        y = self.gln1(self.prelu1(self.conv1(y)))
        y = nn.functional.pad(y, (0, self.dilation * 2)) 
        y = self.gln2(self.prelu2(self.conv2(y)))
        y = self.conv3(y) + x
        return y
