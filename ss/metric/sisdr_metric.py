import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from .base_metric import BaseMetric


class SISDR(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True).to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.sisdr(s1, target)
        except:
            res = torch.tensor([0.0], device=self.device)

        return res
