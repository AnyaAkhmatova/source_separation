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

    def make_mask(self, lens):
        mask = torch.ones(lens.shape[0], int(lens.max().item()), dtype=torch.float32, device=lens.device)
        for i, len in enumerate(lens):
            mask[i, len:] = 0.0
        return mask

    def sisdr(self, pred, target, mask, lens):
        pred = pred * mask
        pred = pred - pred.sum(1, keepdim=True) / lens
        target = target - target.sum(1, keepdim=True) / lens
        pred = pred * mask
        target = target * mask
        num = ((pred * target).sum(1) / ((target * target).sum(1) + self.eps)).unsqueeze(-1) * target 
        denom = num - pred
        result = (10 * torch.log10((num**2).sum(1) / ((denom**2).sum(1) + self.eps) + self.eps)).mean()
        return result
    
    def forward(self, s1, s2, s3, target, lens, logits=None, target_id=None, is_train=True, **batch):
        mask = self.make_mask(lens)
        lens = lens.reshape(-1, 1)
        si_sdr = self.sisdr(s1, target, mask, lens)
        result = (1 - self.alpha - self.beta) * si_sdr
        result += self.alpha * self.sisdr(s2, target, mask, lens)
        result += self.beta * self.sisdr(s3, target, mask, lens)
        result = -result 
        if is_train:
            result += self.gamma * self.ce(logits, target_id) * int(torch.any(target_id != -100).item())
        return result, si_sdr
