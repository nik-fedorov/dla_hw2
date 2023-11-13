import torch
import torch.nn as nn

from .si_sdr_loss import SISDRLoss


class SpExLoss(nn.Module):
    def __init__(self, gamma, si_sdr_params, cross_entropy_params):
        super().__init__()
        self._sdr_loss = SISDRLoss(**si_sdr_params)
        self._ce_loss = torch.nn.CrossEntropyLoss(**cross_entropy_params)
        self.gamma = gamma

    def __call__(self, speaker_logits, speaker_id=None, **batch):
        sdr_loss = self._sdr_loss(**batch)
        if speaker_id is not None:
            ce_loss = self._ce_loss(speaker_logits, speaker_id)
        else:
            ce_loss = 0.0   # on validation and test speaker_id is not added to batch and we compute only SISDRLoss
        return sdr_loss + self.gamma * ce_loss
