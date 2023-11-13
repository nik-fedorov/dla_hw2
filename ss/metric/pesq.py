from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from ss.base.base_metric import BaseMetric
from ss.utils.util import mask_sequences


class PESQMetric(BaseMetric):
    def __init__(self, sr=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(sr, 'wb')

    def __call__(self, short, target, mix_lengths, **batch):
        return self.pesq(mask_sequences(short, mix_lengths).cpu(), target.cpu())
