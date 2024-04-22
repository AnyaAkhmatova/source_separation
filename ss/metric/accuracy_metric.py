import torch

from .base_metric import BaseMetric


class ACC(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __call__(self, logits=None, target_id=None, **kwargs):
        if logits is None:
            return torch.tensor([0.0], device=self.device)
        preds = logits.argmax(-1)
        return (preds == target_id).float().mean()
