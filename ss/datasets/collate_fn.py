import torch


def collate_fn(batch):
    mix_max_len = max([sample["mix"].shape[-1] for sample in batch])
    ref_max_len = max([sample["ref"].shape[-1] for sample in batch])

    mix_tensor = torch.zeros(len(batch), mix_max_len)
    target_tensor = torch.zeros(len(batch), mix_max_len)
    ref_tensor = torch.zeros(len(batch), ref_max_len)
    target_ids = torch.zeros(len(batch), dtype=int)
    mix_lens = torch.zeros(len(batch), dtype=int)

    for i, sample in enumerate(batch):
        mix_tensor[i, :sample["mix"].shape[-1]] = sample["mix"][0]
        target_tensor[i, :sample["target"].shape[-1]] = sample["target"][0]
        ref_tensor[i, :sample["ref"].shape[-1]] = sample["ref"][0]
        target_ids[i] = sample["target_id"]
        mix_lens[i] = sample["mix"].shape[-1]

    return {
        "mix": mix_tensor,
        "target": target_tensor,
        "ref": ref_tensor,
        "target_id": target_ids,
        "lens": mix_lens
    }
