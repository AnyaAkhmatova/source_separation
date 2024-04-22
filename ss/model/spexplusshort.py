import torch.nn as nn

from .tcn_block import TCNBlockRef, TCNBlock
from .resnet_block import ResNetBlock


class SpeechEncoderShort(nn.Module):
    def __init__(self, n_channels, short_kernel):
        super().__init__()
        self.n_channels = n_channels 
        self.short_kernel = short_kernel

        self.encoder_short = nn.Conv1d(1, n_channels, short_kernel, stride=short_kernel//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        init_len = x.shape[-1]
        if (init_len - self.short_kernel) % (self.short_kernel // 2) != 0:
            x = nn.functional.pad(x, (0, 
                (self.short_kernel // 2) - (init_len - self.short_kernel) % (self.short_kernel // 2)
            ))
        x = x.unsqueeze(1)
        y = self.relu(self.encoder_short(x))
        return y, init_len


class SpeakerEncoderShort(nn.Module):
    def __init__(self, in_channels, out_channels, n_resnetblocks, n_speakers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_resnetblocks = n_resnetblocks
        self.n_speakers = n_speakers

        self.ln1 = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.resnetblocks = nn.ModuleList(
            [ResNetBlock(in_channels, in_channels) for _ in range(n_resnetblocks//2)] + \
            [ResNetBlock(in_channels, out_channels) for _ in range(1)] + \
            [ResNetBlock(out_channels, out_channels) for _ in range(n_resnetblocks - n_resnetblocks//2 - 1)]
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.linear = nn.Linear(out_channels, n_speakers)

    def forward(self, x, have_relevant_speakers):
        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)
        for block in self.resnetblocks:
            y = block(y)
        y = self.conv2(y)
        ref_vec = y.mean(2)
        if have_relevant_speakers:
            y = self.linear(ref_vec)
            return ref_vec, y
        return ref_vec


class SpeakerExtractorShort(nn.Module):
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
        for _ in range(n_stacked_tcnblocks):
            self.stacked_tcnblocks.append(nn.ModuleList(
                [TCNBlockRef(n_channels, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal),
                *[TCNBlock(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)]]
            ))
        self.stacked_tcnblocks = nn.ModuleList(self.stacked_tcnblocks)

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x, ref):
        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)
        for i in range(len(self.stacked_tcnblocks)):
            y = self.apply_stacked_tcnblocks(y, ref, self.stacked_tcnblocks[i])
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


class SpeechDecoderShort(nn.Module):
    def __init__(self, n_channels, short_kernel):
        super().__init__()
        self.n_channels = n_channels 
        self.short_kernel = short_kernel

        self.decoder_short = nn.ConvTranspose1d(n_channels, 1, short_kernel, stride=short_kernel//2)

    def forward(self, encs1, init_len):
        s1 = self.decoder_short(encs1)[:, 0, :init_len]
        return s1


class SpexPlusShort(nn.Module):
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
        self.speaker_extractor = SpeakerExtractorShort(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, causal)
        self.speech_decoder = SpeechDecoderShort(n_channels, short_kernel)

    def forward(self, x, ref, have_relevant_speakers):
        mix_encs, mix_init_len = self.speech_encoder(x)

        ref_encs, _ = self.speech_encoder(ref)
        results = self.speaker_encoder(ref_encs, have_relevant_speakers)
        if have_relevant_speakers:
            ref_vec, speaker_logits = results
        else:
            ref_vec = results

        ref_vec = ref_vec.repeat(x.shape[0] // ref_vec.shape[0], 1)

        encs1 = self.speaker_extractor(mix_encs, ref_vec)
        s1 = self.speech_decoder(encs1, mix_init_len)
        if have_relevant_speakers:
            return s1, speaker_logits
        return s1