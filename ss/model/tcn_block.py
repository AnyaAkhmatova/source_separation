import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, 
                 n_channels, 
                 eps=1e-6):
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
    

class CumulativeLayerNorm(nn.Module):
    def __init__(self, 
                 n_channels, 
                 eps=1e-6):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(1, n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))            

    def forward(self, x):
        mean = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        std = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        for i in range(x.shape[-1]):
            mean[:, :, i: i + 1] = x[:, :, : i + 1].mean(dim=(1, 2)).reshape(-1, 1, 1)
            std[:, :, i: i + 1] = ((((x[:, :, : i + 1] - mean[:, :, i: i + 1])**2).mean(dim=(1, 2)) + self.eps)**0.5).reshape(-1, 1, 1)
        y = (x - mean) / std
        y = self.weight * y + self.bias
        return y


class TCNBlock(nn.Module):
    def __init__(self, 
                 n_channels, 
                 hidden_channels,
                 dilation, 
                 causal=False):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.causal = causal

        self.conv1 = nn.Conv1d(n_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(hidden_channels) if not causal else CumulativeLayerNorm(hidden_channels)
        self.deconv = nn.Sequential(        
            nn.Conv1d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=3, 
                      dilation=dilation, 
                      groups=hidden_channels),
            nn.Conv1d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=1)
        )
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(hidden_channels) if not causal else CumulativeLayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, n_channels, kernel_size=1)

    def forward(self, x):
        y = self.gln1(self.prelu1(self.conv1(x)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation)) 
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0)) 
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y) + x
        return y


class TCNBlockRef(nn.Module):
    def __init__(self, 
                 n_channels, 
                 hidden_channels,
                 dilation, 
                 ref_dim, 
                 causal=False):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.ref_dim = ref_dim
        self.causal = causal

        self.conv1 = nn.Conv1d(n_channels + ref_dim, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(hidden_channels) if not causal else CumulativeLayerNorm(hidden_channels)
        self.deconv = nn.Sequential(        
            nn.Conv1d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=3, 
                      dilation=dilation, 
                      groups=hidden_channels),
            nn.Conv1d(hidden_channels, 
                      hidden_channels, 
                      kernel_size=1)
        )
        self.prelu2 = nn.PReLU()
        self.gln2 = GlobalLayerNorm(hidden_channels) if not causal else CumulativeLayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, n_channels, kernel_size=1)

    def forward(self, x, ref):
        ref = ref.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = torch.cat([x, ref], 1)
        y = self.gln1(self.prelu1(self.conv1(y)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation)) 
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0)) 
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y) + x
        return y
