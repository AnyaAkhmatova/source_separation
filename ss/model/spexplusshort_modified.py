import torch
import torch.nn as nn

from .tcn_block import TCNBlockRef, TCNBlock
from .resnet_block import ResNetBlock
from .spexplusshort import SpeechEncoderShort, SpeakerEncoderShort, SpeechDecoderShort


class FMS(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y, _ = torch.max(x, dim=-1)
        y = self.sigmoid(self.fc(y))
        y = y.unsqueeze(-1)
        y = x * y + y
        return y


class SpeakerExtractorShortMod(nn.Module):
    def __init__(self, n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, causal=False):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_stacked_tcnblocks = n_stacked_tcnblocks
        self.n_tcnblocks = n_tcnblocks
        self.causal = causal

        self.ln1 = nn.LayerNorm(n_channels)
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        self.stacked_tcnblocks = []
        self.fmss = []
        for _ in range(n_stacked_tcnblocks):
            self.stacked_tcnblocks.append(nn.ModuleList(
                [TCNBlockRef(n_channels, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal),
                *[TCNBlock(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)]]
            ))
            self.fmss.append(FMS(n_channels, n_channels))
        self.stacked_tcnblocks = nn.ModuleList(self.stacked_tcnblocks)
        self.fmss = nn.ModuleList(self.fmss)

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, ref):
        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)
        res = None
        for i in range(len(self.stacked_tcnblocks)):
            y = self.apply_stacked_tcnblocks(y, ref, self.stacked_tcnblocks[i])
            y = self.fmss[i](y)
            if i == 0:
                res = y
            else:
                res = res + y
        y = res
        mask1 = self.relu(self.mask1(y))
        encs1 = x * mask1
        return encs1

    def apply_stacked_tcnblocks(self, y, ref, stacked_tcnblocks):
        for i in range(len(stacked_tcnblocks)):
            if i == 0:
                y = stacked_tcnblocks[i](y, ref)
            else:
                y = stacked_tcnblocks[i](y)
        return y


class SpexPlusShortMod(nn.Module):
    def __init__(self, 
                 n_channels, 
                 hidden_channels,
                 out_channels,
                 short_kernel, 
                 n_resnetblocks,
                 n_speakers,
                 n_stacked_tcnblocks,
                 n_tcnblocks, 
                 causal=False):
        super().__init__()
        self.causal = causal

        self.speech_encoder = SpeechEncoderShort(n_channels, short_kernel)
        self.speaker_encoder = SpeakerEncoderShort(n_channels, out_channels, n_resnetblocks, n_speakers)
        self.speaker_extractor = SpeakerExtractorShortMod(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, causal)
        self.speech_decoder = SpeechDecoderShort(n_channels, short_kernel)

    def forward(self, x, ref):
        mix_encs, mix_init_len = self.speech_encoder(x)

        ref_encs, _ = self.speech_encoder(ref)
        if self.training:
            ref_vec, speaker_logits = self.speaker_encoder(ref_encs)
        else:
            ref_vec = self.speaker_encoder(ref_encs)

        ref_vec = ref_vec.repeat(x.shape[0] // ref_vec.shape[0], 1)

        encs1 = self.speaker_extractor(mix_encs, ref_vec)
        s1 = self.speech_decoder(encs1, mix_init_len)
        if self.training:
            return s1, speaker_logits
        return s1