import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from .base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, device, fs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.stoi = ShortTimeObjectiveIntelligibility(fs=fs, extended=False).to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.stoi(s1, target).to(self.device)
        except:
            res = torch.tensor([0.0], device=self.device)

        return res
