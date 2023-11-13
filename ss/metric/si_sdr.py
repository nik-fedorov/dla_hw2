from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from ss.base.base_metric import BaseMetric
from ss.utils.util import mask_sequences


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, short, target, mix_lengths, **kwargs):
        return self.si_sdr(mask_sequences(short, mix_lengths).cpu(), target.cpu())
