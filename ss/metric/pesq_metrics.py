import torch

from .base_metric import BaseMetric
from .hifi_pesq_metrics import composite_eval


class CompositeMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __call__(self, s1, target, **kwargs):
        s1 = s1.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        results = composite_eval(target, s1)
        for key, value in results.items():
            results[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        return results

