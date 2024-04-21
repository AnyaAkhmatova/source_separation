import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn_block import TCNBlockRef, TCNBlock

from .spexplusshort import SpeechEncoderShort, SpeakerEncoderShort, SpeechDecoderShort


class SpeakerExtractorShortSpecialTokens(nn.Module):
    def __init__(self, n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                 n_tokens=None, causal=False):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_stacked_tcnblocks = n_stacked_tcnblocks
        self.n_tcnblocks = n_tcnblocks

        self.n_tokens = n_tokens
        if self.n_tokens == None:
            self.n_tokens = 2**n_tcnblocks
        self.causal = causal

        self.ln1 = nn.LayerNorm(n_channels)
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        self.stacked_tcnblocks = []
        for _ in range(n_stacked_tcnblocks):
            self.stacked_tcnblocks.append(
                nn.ModuleList(
                    [TCNBlockRef(n_channels, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal),
                     *[TCNBlock(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)]]
                )
            )
        self.stacked_tcnblocks = nn.ModuleList(self.stacked_tcnblocks)

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, ref, tokens=None):
        if tokens is None:
            tokens = torch.zeros((x.shape[0], x.shape[-2], self.n_tokens), dtype=x.dtype, device=x.device)

        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)

        y = torch.cat([tokens, y, tokens], dim=-1)
        for i in range(len(self.stacked_tcnblocks)):
            y = self.apply_stacked_tcn_blocks(self.stacked_tcnblocks[i], y, ref)
        new_tokens = y[:, :, y.shape[-1] - self.n_tokens:]
        y = y[:, :, self.n_tokens: y.shape[-1] - self.n_tokens]

        mask1 = self.relu(self.mask1(y))
        encs1 = x * mask1
        return encs1, new_tokens

    def apply_stacked_tcn_blocks(self, stacked_tcn_blocks, x, ref):
        y = stacked_tcn_blocks[0](x, ref)
        for i in range(1, len(stacked_tcn_blocks)):
            y = stacked_tcn_blocks[i](y)
        return y


class SpexPlusShortSpecialTokensModel(nn.Module):
    def __init__(self,
                 n_channels,
                 hidden_channels,
                 out_channels,
                 short_kernel,
                 n_resnetblocks,
                 n_speakers,
                 n_stacked_tcnblocks,
                 n_tcnblocks,
                 n_tokens=None,
                 causal=False):
        super().__init__()
        self.causal = causal

        self.speech_encoder = SpeechEncoderShort(n_channels, short_kernel)
        self.speaker_encoder = SpeakerEncoderShort(n_channels, out_channels, n_resnetblocks, n_speakers)
        self.speaker_extractor = SpeakerExtractorShortSpecialTokens(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                                                                    n_tokens, causal)
        self.speech_decoder = SpeechDecoderShort(n_channels, short_kernel)

    def forward_chunk(self, x, ref_vec, tokens):
        mix_encs, mix_init_len = self.speech_encoder(x)
        encs1, new_tokens = self.speaker_extractor(mix_encs, ref_vec, tokens)
        s1 = self.speech_decoder(encs1, mix_init_len)
        return s1, new_tokens

    def forward(self, x, ref=None, ref_vec=None, tokens=None, one_chunk=False):
        # training: work with all chunks inside model.forward in oreder to
        # use ddp backward, otherwise inplace modification error (because of reference processing)
        # validating/testing: work with all chunks inside model.forward as
        # we still have full access to the audios and can simplify training code
        # inferencing: need to use one_chunk=True and specialized inferencer
        # to follow sequential chunk processing logic outside model.forward (will be written later)
        if not one_chunk: # have x and ref
            batch_size = ref.shape[0]
            n_chunks = x.shape[0] // ref.shape[0]
            ref_encs, _ = self.speech_encoder(ref)
            if self.training:
                ref_vec, speaker_logits = self.speaker_encoder(ref_encs)
            else:
                ref_vec = self.speaker_encoder(ref_encs)
            results = []
            tokens = None
            for i in range(n_chunks):
                chunk = x[i * batch_size: (i + 1) * batch_size, :]
                res, tokens = self.forward_chunk(chunk, ref_vec, tokens)
                results.append(res)
            s1 = torch.cat(results, dim=0)
            if self.training:
                return s1, speaker_logits
            return s1
        # inference
        if ref_vec is None: # have x and ref
            ref_encs, _ = self.speech_encoder(ref)
            if self.training:
                ref_vec, speaker_logits = self.speaker_encoder(ref_encs)
            else:
                ref_vec = self.speaker_encoder(ref_encs)
            s1, new_tokens = self.forward_chunk(x, ref_vec, None)
            if self.training:
                return s1, new_tokens, ref_vec, speaker_logits
            return s1, new_tokens, ref_vec
        # have x, ref_vec and tokens
        s1, new_tokens = self.forward_chunk(x, ref_vec, tokens)
        return s1, new_tokens
    
