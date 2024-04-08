import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from .base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.pesq(s1, target)
        except:
            res = torch.tensor([0.0], device=self.device)

        return res
