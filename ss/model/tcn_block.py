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
    

class ChannelLayerNorm(nn.Module):
    def __init__(self, 
                 n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.ln = nn.LayerNorm(n_channels)

    def forward(self, x):
        y = self.ln(x.transpose(1, 2)).transpose(1, 2)
        return y


class TCNBlock(nn.Module):
    def __init__(self, 
                 n_channels, 
                 hidden_channels,
                 dilation, 
                 causal=False,
                 out_channels=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.causal = causal
        self.out_channels = out_channels
        if out_channels is None:
            self.out_channels = n_channels
            out_channels = n_channels

        self.conv1 = nn.Conv1d(n_channels, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(hidden_channels) if not causal else ChannelLayerNorm(hidden_channels)
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
        self.gln2 = GlobalLayerNorm(hidden_channels) if not causal else ChannelLayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        if out_channels != n_channels:
            self.downsample = nn.Conv1d(n_channels, out_channels, kernel_size=1)

    def forward(self, x):
        y = self.gln1(self.prelu1(self.conv1(x)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation)) 
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0)) 
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y)
        if self.n_channels == self.out_channels:
            y = y + x
        else:
            y = y + self.downsample(x)
        return y


class TCNBlockRef(nn.Module):
    def __init__(self, 
                 n_channels, 
                 hidden_channels,
                 dilation, 
                 ref_dim, 
                 causal=False, 
                 out_channels=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        self.ref_dim = ref_dim
        self.causal = causal
        self.out_channels = out_channels
        if out_channels is None:
            self.out_channels = n_channels
            out_channels = n_channels

        self.conv1 = nn.Conv1d(n_channels + ref_dim, hidden_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = GlobalLayerNorm(hidden_channels) if not causal else ChannelLayerNorm(hidden_channels)
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
        self.gln2 = GlobalLayerNorm(hidden_channels) if not causal else ChannelLayerNorm(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        if out_channels != n_channels:
            self.downsample = nn.Conv1d(n_channels, out_channels, kernel_size=1)

    def forward(self, x, ref):
        ref = ref.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = torch.cat([x, ref], 1)
        y = self.gln1(self.prelu1(self.conv1(y)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation)) 
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0)) 
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y)
        if self.n_channels == self.out_channels:
            y = y + x
        else:
            y = y + self.downsample(x)
        return y


class TCNBlockRNN(TCNBlock):
    def __init__(self,
                 n_channels,
                 hidden_channels,
                 dilation,
                 causal=False,
                 out_channels=None):
        super().__init__(n_channels,
                         hidden_channels,
                         dilation,
                         causal, 
                         out_channels)

    def forward(self, x, memory):
        memory = memory.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = self.gln1(self.prelu1(self.conv1(x + memory)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation))
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0))
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y)
        if self.n_channels == self.out_channels:
            y = y + x
        else:
            y = y + self.downsample(x)
        return y


class TCNBlockRefRNN(TCNBlockRef):
    def __init__(self,
                 n_channels,
                 hidden_channels,
                 dilation,
                 ref_dim,
                 causal=False, 
                 out_channels=None):
        super().__init__(n_channels,
                         hidden_channels,
                         dilation,
                         ref_dim,
                         causal, 
                         out_channels)

    def forward(self, x, ref, memory):
        ref = ref.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        memory = memory.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        y = torch.cat([x + memory, ref], 1)
        y = self.gln1(self.prelu1(self.conv1(y)))
        if not self.causal:
            y = nn.functional.pad(y, (self.dilation, self.dilation))
        else:
            y = nn.functional.pad(y, (self.dilation * 2, 0))
        y = self.gln2(self.prelu2(self.deconv(y)))
        y = self.conv2(y)
        if self.n_channels == self.out_channels:
            y = y + x
        else:
            y = y + self.downsample(x)
        return y
