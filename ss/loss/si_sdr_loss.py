import torch
from torch import Tensor
import torch.nn as nn

from ss.utils.util import mask_sequences


class SISDRLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def _si_sdr(self, pred, target, eps=1e-6):
        alpha = (torch.sum(pred * target, dim=-1, keepdim=True) + eps) / \
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
        target_scaled = alpha * target
        val = (torch.sum(target_scaled ** 2, dim=-1) + eps) / (torch.sum((target_scaled - pred) ** 2, dim=-1) + eps)
        return 20 * torch.log10(val)

    def forward(self, short, middle, long, target, mix_lengths, **batch) -> Tensor:
        sdr_short = torch.sum(self._si_sdr(mask_sequences(short, mix_lengths), target))
        sdr_middle = torch.sum(self._si_sdr(mask_sequences(middle, mix_lengths), target))
        sdr_long = torch.sum(self._si_sdr(mask_sequences(long, mix_lengths), target))
        return - ((1 - self.alpha - self.beta) * sdr_short + self.alpha * sdr_middle + self.beta * sdr_long) / short.size(0)
