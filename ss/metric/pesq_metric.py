from .base_metric import BaseMetric

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQ(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to(device)

    def __call__(self, s1, target, **kwargs):
        res = 0
        try:
            res = self.pesq(s1, target).item()
        except:
            res = 0

        return res
