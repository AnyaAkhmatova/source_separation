import torch
from torch import nn


def overlap_add_half(chunks, n_chunks, window, step):
    batch_size = chunks.shape[0] // n_chunks
    length = window

    chunks = chunks.reshape(n_chunks, batch_size, length).permute(1, 0, 2)
    chunks = chunks.reshape(batch_size, -1, step)
    first_part = chunks[:, :1, :]
    last_part = chunks[:, -1:, :]
    chunks = chunks[:, 1: -1, :].reshape(batch_size, -1, window)

    chunks = (chunks @ torch.eye(step, dtype=chunks.dtype, device=chunks.device).repeat(2, 1).unsqueeze(0)) / 2
    chunks = torch.cat([first_part, chunks, last_part], dim=1).reshape(batch_size, -1)
    return chunks

def overlap_add_nonintersec(chunks, n_chunks, window, step):
    batch_size = chunks.shape[0] // n_chunks
    length = window

    chunks = chunks.reshape(n_chunks, batch_size, length).permute(1, 0, 2).reshape(batch_size, -1)
    return chunks


overlap_add_methods = {
    'half': overlap_add_half,
    'nonintersec': overlap_add_nonintersec
}


class Streamer:
    def __init__(self, chunk_window, chunk_step, overlap_add_method):
        self.chunk_window = chunk_window
        self.chunk_step = chunk_step
        self.overlap_add_method_name = overlap_add_method
        assert overlap_add_method in ['half', 'nonintersec'], "overlap_add_method not supported"
        if overlap_add_method == 'half':
            assert chunk_step == chunk_window // 2
        else:
            assert chunk_step == chunk_window
        self.overlap_add_method = overlap_add_methods[overlap_add_method]

    def make_chunks(self, mix):
        if (mix.shape[-1] - self.chunk_window) % self.chunk_step != 0:
            mix = nn.functional.pad(mix, (0, 
                self.chunk_step - (mix.shape[-1] - self.chunk_window) % self.chunk_step
            ))
        n_chunks = (mix.shape[-1] - self.chunk_window) // self.chunk_step + 1
        if self.overlap_add_method_name == 'half':
            mix_chunks = mix.reshape(mix.shape[0], n_chunks + 1, self.chunk_step)
            mix_chunks = torch.cat([mix_chunks[:, : -1, :], mix_chunks[:, 1:, :]], dim=2).permute(1, 0, 2).reshape(-1, self.chunk_window)
        else:
            mix_chunks = mix.reshape(mix.shape[0], n_chunks, self.chunk_window).permute(1, 0, 2).reshape(-1, self.chunk_window)
        return mix_chunks, n_chunks

    def apply_overlap_add_method(self, chunks, n_chunks):
        assert n_chunks != 0, "no chunks to overlap_add"
        return self.overlap_add_method(chunks, n_chunks, self.chunk_window, self.chunk_step)

