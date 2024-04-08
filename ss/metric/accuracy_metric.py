from .base_metric import BaseMetric


class ACC(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, target_id, **kwargs):
        preds = logits.argmax(-1)
        return (preds == target_id).float().mean()
