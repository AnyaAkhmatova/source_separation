import torch
from torchmetrics.audio import SignalNoiseRatio

from .base_metric import BaseMetric


class SNR(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.snr = SignalNoiseRatio(zero_mean=True).to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.snr(s1, target).to(self.device)
        except:
            res = torch.tensor([0.0], device=self.device)

        return res
