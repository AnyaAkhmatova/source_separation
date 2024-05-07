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


def inference_collate_fn(batch):
    if batch.get("target", None) is None:
        return {
            "mix": batch[0]["mix"],
            "ref": batch[0]["ref"],
            "lens": torch.tensor(batch[0]["mix"].shape[-1], dtype=int),
            "pred_name": [batch[0]["pred_name"]]
        }
    
    return {
        "mix": batch[0]["mix"],
        "ref": batch[0]["ref"],
        "target": batch[0]["target"],
        "lens": torch.tensor(batch[0]["mix"].shape[-1], dtype=int),
        "pred_name": [batch[0]["pred_name"]]
    }
