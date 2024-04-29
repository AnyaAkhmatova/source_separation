import torch

from .base_metric import BaseMetric
from .hifi_pesq_metrics import composite_eval


class CompositeMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __call__(self, s1, target, **kwargs):
        assert target.shape[0] == 1, "batch_size must be 1"

        s1 = s1.detach().cpu().reshape(-1).numpy()
        target = target.detach().cpu().reshape(-1).numpy()
        results = composite_eval(target, s1)
        for key, value in results.items():
            results[key] = torch.tensor([value], dtype=torch.float32, device=self.device)
        return results

