import torch
import torch.nn as nn


class SpexPlusLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, gamma=0.5, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def sisdr(self, pred, target):
        pred = pred - pred.mean(1, keepdim=True)
        target = target - target.mean(1, keepdim=True)
        num = ((pred * target).sum(1) / ((target * target).sum(1) + self.eps)).unsqueeze(-1) * target 
        denom = num - pred
        result = (10 * torch.log10((num**2).sum(1) / ((denom**2).sum(1) + self.eps) + self.eps)).mean()
        return result
    
    def forward(self, s1, s2, s3, target, logits, target_id=None, is_train=True, **batch):
        result = (1 - self.alpha - self.beta) * self.sisdr(s1, target)
        result += self.alpha * self.sisdr(s2, target)
        result += self.beta * self.sisdr(s3, target)
        result = -result 
        if is_train:
            result += self.gamma * self.ce(logits, target_id)
        return result
