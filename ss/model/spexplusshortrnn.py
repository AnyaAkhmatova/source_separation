import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn_block import TCNBlockRef, TCNBlock, TCNBlockRefRNN, TCNBlockRNN

from .spexplusshort import SpeechEncoderShort, SpeakerEncoderShort, SpeechDecoderShort


class SpeakerExtractorShortRNN(nn.Module):
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
        for _ in range(0, n_stacked_tcnblocks - 1):
            self.stacked_tcnblocks.append(
                nn.ModuleList(
                    [TCNBlockRef(n_channels, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal),
                     *[TCNBlock(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)]]
                )
            )
        self.stacked_tcnblocks.append(
            nn.ModuleList(
                    [TCNBlockRefRNN(n_channels, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal),
                     *[TCNBlockRNN(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)],
                     nn.Conv1d(n_channels * (n_tcnblocks - 2), n_channels, kernel_size=1)]
            )            
        )
        self.stacked_tcnblocks = nn.ModuleList(self.stacked_tcnblocks)

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x, ref, memory=None):
        if memory is None:
            memory = torch.zeros(x.shape[0], self.n_channels, dtype=x.dtype, device=x.device)
        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)
        for i in range(0, len(self.stacked_tcnblocks) - 1):
            y = self.apply_stacked_tcn_blocks(self.stacked_tcnblocks[i], y, ref)        
        y, new_memory = self.apply_last_stacked_tcn_blocks(self.stacked_tcnblocks[-1], y, ref, memory)
        mask1 = self.relu(self.mask1(y))
        encs1 = x * mask1
        return encs1, new_memory

    def apply_stacked_tcn_blocks(self, stacked_tcn_blocks, x, ref):
        y = stacked_tcn_blocks[0](x, ref)
        for i in range(1, len(stacked_tcn_blocks)):
            y = stacked_tcn_blocks[i](y)
        return y
    
    def apply_last_stacked_tcn_blocks(self, stacked_tcn_blocks, x, ref, memory):
        y = stacked_tcn_blocks[0](x, ref, memory)
        results = []
        for i in range(1, len(stacked_tcn_blocks) - 2):
            results.append(stacked_tcn_blocks[i](y, memory))
        f = F.adaptive_avg_pool1d(stacked_tcn_blocks[-1](torch.cat(results, dim=1)), 1).squeeze(-1)
        o = stacked_tcn_blocks[-2](results[-1], memory)
        g = F.adaptive_avg_pool1d(o, 1).squeeze(-1)
        sig_f = F.sigmoid(f)
        new_memory = sig_f * g + (1 - sig_f) * memory
        return o, new_memory


class SpexPlusShortRNN(nn.Module):
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
        self.speaker_extractor = SpeakerExtractorShortRNN(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, causal)
        self.speech_decoder = SpeechDecoderShort(n_channels, short_kernel)

    def forward_chunk(self, x, ref_vec, memory):
        mix_encs, mix_init_len = self.speech_encoder(x)
        encs1, new_memory = self.speaker_extractor(mix_encs, ref_vec, memory)
        s1 = self.speech_decoder(encs1, mix_init_len)
        return s1, new_memory

    def forward(self, x, ref=None, ref_vec=None, memory=None, one_chunk=False):
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
            ref_vec, speaker_logits = self.speaker_encoder(ref_encs)
            results = []
            memory = None
            for i in range(n_chunks):
                chunk = x[i * batch_size: (i + 1) * batch_size, :]
                res, memory = self.forward_chunk(chunk, ref_vec, memory)
                results.append(res)           
            s1 = torch.cat(results, dim=0)
            return s1, speaker_logits
        # inference
        if ref_vec is None: # have x and ref
            ref_encs, _ = self.speech_encoder(ref)
            ref_vec, speaker_logits = self.speaker_encoder(ref_encs)
            s1, new_memory = self.forward_chunk(x, ref_vec, None)
            return s1, new_memory, ref_vec, speaker_logits
        # have x, ref_vec and memory
        s1, new_memory = self.forward_chunk(x, ref_vec, memory)
        return s1, new_memory
