import logging
from pathlib import Path

import torchaudio

from ss.base.base_mixture_dataset import BaseMixtureDataset

logger = logging.getLogger(__name__)


class CustomMixtureDataset(BaseMixtureDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            for audio in ['mix', 'ref', 'target']:
                assert audio in entry
                assert Path(entry[audio]).exists(), f"Path {entry[audio]} doesn't exist"
                entry[audio] = str(Path(entry[audio]).absolute().resolve())
        super().__init__(index, *args, **kwargs)

    @property
    def num_speakers(self):
        return None
