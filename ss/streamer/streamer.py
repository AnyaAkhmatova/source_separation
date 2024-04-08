import torch
from torch import nn


def overlap_add_half(chunks, window, step):
    assert step == window // 2
    wave = torch.zeros((chunks[0].shape[0], (len(chunks) - 1) * step + window), dtype=chunks[0].dtype, device=chunks[0].device)
    for i, chunk in enumerate(chunks):
        wave[:, i * step: i * step + window] = wave[:, i * step: i * step + window] + chunk
    wave[:, step: wave.shape[-1] - step] = wave[:, step: wave.shape[-1] - step] / 2
    return wave

def overlap_add_nonintersec(chunks, window, step):
    assert step == window
    wave = torch.zeros((chunks[0].shape[0], len(chunks) * window), dtype=chunks[0].dtype, device=chunks[0].device)
    for i, chunk in enumerate(chunks):
        wave[:, i * window: (i + 1) * window] = chunk
    return wave

def overlap_add_sin(chunks, window, step):
    assert step == window // 2
    wave = torch.zeros((chunks[0].shape[0], (len(chunks) - 1) * step + window), dtype=chunks[0].dtype, device=chunks[0].device)
    coef = torch.sin(torch.arange(step) / (step - 1) * torch.acos(torch.zeros(1)).item()).to(wave.device)
    for i, chunk in enumerate(chunks):
        if i == 0:
            wave[:, i * step: i * step + window] = chunk
        else:
            wave[:, i * step: (i + 1) * step] = wave[:, i * step: (i + 1) * step] * (1 - coef) + chunk[: step] * coef
            wave[:, (i + 1) * step: i * step + window] = chunk[step:]
    return wave


overlap_add_methods = {
    'half': overlap_add_half,
    'nonintersec': overlap_add_nonintersec,
    'sin': overlap_add_sin
}


class Streamer:
    def __init__(self, chunk_window, chunk_step, overlap_add_method):
        self.chunk_window = chunk_window
        self.chunk_step = chunk_step
        assert chunk_step <= chunk_window 
        self.overlap_add_method = overlap_add_methods[overlap_add_method]

    def iterate_through(self, mix):
        if (mix.shape[-1] - self.chunk_window) % self.chunk_step != 0:
            mix = nn.functional.pad(mix, (0, 
                self.chunk_step - (mix.shape[-1] - self.chunk_window) % self.chunk_step
            ))

        for i in range(0, (mix.shape[-1] - self.chunk_window) // self.chunk_step + 1):
            chunk_mix = mix[:, i * self.chunk_step: i * self.chunk_step + self.chunk_window]
            yield chunk_mix

    def apply_overlap_add_method(self, chunks):
        assert len(chunks) != 0, "no chunks to overlap_add"
        return self.overlap_add_method(chunks, self.chunk_window, self.chunk_step)

