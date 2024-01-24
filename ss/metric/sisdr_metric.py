from .base_metric import BaseMetric

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SISDR(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sisdr = ScaleInvariantSignalDistortionRatio(zero_mean=True).to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.sisdr(s1, target).item()
        except:
            res = 0

        return res
