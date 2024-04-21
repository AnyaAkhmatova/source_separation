import torch
import torch.nn as nn
import torch.nn.functional as F

from .tcn_block import TCNBlockRef, TCNBlock

from .spexplusshort import SpeechEncoderShort, SpeakerEncoderShort, SpeechDecoderShort


class SpeakerExtractorShortCacheTime(nn.Module):
    def __init__(self, n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                 streamer_type, causal=False, memory_size=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_stacked_tcnblocks = n_stacked_tcnblocks
        self.n_tcnblocks = n_tcnblocks

        self.streamer_type = streamer_type
        self.causal = causal
        self.memory_size = memory_size
        if memory_size is None:
            self.memory_size = 2**n_tcnblocks if not causal else 2**(n_tcnblocks + 1)
            memory_size = self.memory_size

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

        self.cache = []

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def reset_cache(self):
        self.cache = []

    def forward(self, x, ref):
        if len(self.cache) == 0:
            self.cache = [torch.zeros((x.shape[0], x.shape[-2], self.memory_size), dtype=x.dtype, device=x.device) for _ in range(len(self.stacked_tcnblocks))]
    
        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)

        y = torch.cat([torch.zeros((x.shape[0], x.shape[-2], self.memory_size), dtype=x.dtype, device=x.device), y], dim=-1)
        for i in range(len(self.stacked_tcnblocks)):
            y[:, :, : self.memory_size] = self.cache[i]
            make_memory_from = y[:, :, self.memory_size:]
            make_memory_from = make_memory_from[:, :, : make_memory_from.shape[-1]//2] if self.streamer_type == "half" else make_memory_from[:, :, make_memory_from.shape[-1]//2:]
            self.cache[i] = (make_memory_from[:, :, make_memory_from.shape[-1] - self.memory_size:]).detach()
            y = self.apply_stacked_tcn_blocks(self.stacked_tcnblocks[i], y, ref)
        y = y[:, :, self.memory_size:]

        mask1 = self.relu(self.mask1(y))
        encs1 = x * mask1
        return encs1

    def apply_stacked_tcn_blocks(self, stacked_tcn_blocks, x, ref):
        y = stacked_tcn_blocks[0](x, ref)
        for i in range(1, len(stacked_tcn_blocks)):
            y = stacked_tcn_blocks[i](y)
        return y


class SpeakerExtractorShortCacheChannels(nn.Module):
    def __init__(self, n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                 causal=False, memory_size=None):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_stacked_tcnblocks = n_stacked_tcnblocks
        self.n_tcnblocks = n_tcnblocks

        self.causal = causal
        self.memory_size = memory_size
        if memory_size is None:
            self.memory_size = 2**n_tcnblocks * n_tcnblocks
            memory_size = self.memory_size

        self.ln1 = nn.LayerNorm(n_channels)
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        self.stacked_tcnblocks = []
        for _ in range(n_stacked_tcnblocks):
            self.stacked_tcnblocks.append(
                nn.ModuleList(
                    [TCNBlockRef(n_channels + self.memory_size, hidden_channels, dilation=2**0, ref_dim=out_channels, causal=causal, out_channels=n_channels),
                     *[TCNBlock(n_channels, hidden_channels, dilation=2**i, causal=causal) for i in range(1, n_tcnblocks)]]
                )
            )
        self.stacked_tcnblocks = nn.ModuleList(self.stacked_tcnblocks)

        self.cache = []

        self.mask1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def reset_cache(self):
        self.cache = []

    def forward(self, x, ref):
        if len(self.cache) == 0:
            self.cache = [torch.zeros((x.shape[0], self.memory_size, x.shape[-1]), dtype=x.dtype, device=x.device) for _ in range(len(self.stacked_tcnblocks))]

        y = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        y = self.conv1(y)

        for i in range(len(self.stacked_tcnblocks)):
            y = torch.cat([y, self.cache[i]], dim=-2)
            make_memory_from = y[:, : y.shape[-2] - self.memory_size, :]
            self.cache[i] = (make_memory_from[:, : self.memory_size, :]).detach()
            y = self.apply_stacked_tcn_blocks(self.stacked_tcnblocks[i], y, ref)
        
        mask1 = self.relu(self.mask1(y))
        encs1 = x * mask1
        return encs1

    def apply_stacked_tcn_blocks(self, stacked_tcn_blocks, x, ref):
        y = stacked_tcn_blocks[0](x, ref)
        for i in range(1, len(stacked_tcn_blocks)):
            y = stacked_tcn_blocks[i](y)
        return y


class SpexPlusShortCacheModel(nn.Module):
    def __init__(self,
                 n_channels,
                 hidden_channels,
                 out_channels,
                 short_kernel,
                 n_resnetblocks,
                 n_speakers,
                 n_stacked_tcnblocks,
                 n_tcnblocks,
                 cache_type,
                 streamer_type=None,
                 causal=False, 
                 memory_size=None):
        super().__init__()
        self.causal = causal

        self.speech_encoder = SpeechEncoderShort(n_channels, short_kernel)
        self.speaker_encoder = SpeakerEncoderShort(n_channels, out_channels, n_resnetblocks, n_speakers)
        assert cache_type in ["time", "channels"], "cache_type not implemented"
        if cache_type == "time":
            assert streamer_type in ["half", "nonintersec"], "streamer_type not implemented"
            self.speaker_extractor = SpeakerExtractorShortCacheTime(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                                                                    streamer_type, causal, memory_size)
        else:
            self.speaker_extractor = SpeakerExtractorShortCacheChannels(n_channels, hidden_channels, out_channels, n_stacked_tcnblocks, n_tcnblocks, 
                                                                        causal, memory_size)
        self.speech_decoder = SpeechDecoderShort(n_channels, short_kernel)

    def forward_chunk(self, x, ref_vec):
        mix_encs, mix_init_len = self.speech_encoder(x)
        encs1 = self.speaker_extractor(mix_encs, ref_vec)
        s1 = self.speech_decoder(encs1, mix_init_len)
        return s1

    def forward(self, x, ref=None, ref_vec=None, one_chunk=False):
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
            self.speaker_extractor.reset_cache()
            for i in range(n_chunks):
                chunk = x[i * batch_size: (i + 1) * batch_size, :]
                res = self.forward_chunk(chunk, ref_vec)
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
            self.speaker_extractor.reset_cache()
            s1 = self.forward_chunk(x, ref_vec)
            if self.training:
                return s1, ref_vec, speaker_logits
            return s1, ref_vec
        # have x, ref_vec
        s1 = self.forward_chunk(x, ref_vec)
        return s1
    
