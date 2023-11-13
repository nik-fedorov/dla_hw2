import logging
from pathlib import Path

from ss.datasets.custom_mixture_dataset import CustomMixtureDataset

logger = logging.getLogger(__name__)


class CustomDirMixtureDataset(CustomMixtureDataset):
    def __init__(self, mix_dir, ref_dir, target_dir=None, *args, **kwargs):
        data = []
        for path in Path(mix_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry['mix'] = str(path)
                entry['ref'] = str(Path(ref_dir) / (path.stem.replace('mixed', 'ref') + '.wav'))

                if target_dir and Path(target_dir).exists():
                    target_path = Path(target_dir) / (path.stem.replace('mixed', 'target') + '.wav')
                    if target_path.exists():
                        entry['target'] = str(target_path)
            if len(entry) >= 2:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
