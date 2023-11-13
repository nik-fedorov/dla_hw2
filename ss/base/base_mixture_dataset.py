import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from ss.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseMixtureDataset(Dataset):
    def __init__(
            self,
            index,  # list of {'mix': path1, 'ref': path2, 'target': path3, (optionally) 'speaker_id': int}
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None,
            max_audio_length=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        # self._assert_index_is_valid(index)
        # index = self._filter_records_from_dataset(index, max_audio_length, limit)

        # index = self._sort_index(index)
        self._index: List[dict] = index

    @property
    def num_speakers(self):
        raise NotImplementedError()

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        result = {}
        if 'speaker_id' in data_dict:
            result['speaker_id'] = data_dict['speaker_id']
        for audio in ['mix', 'ref', 'target']:
            audio_wave = self.load_audio(data_dict[audio])
            if audio in ['mix', 'ref'] and self.wave_augs is not None:
                with torch.no_grad():
                    audio_wave = self.wave_augs(audio_wave)
            result[audio] = audio_wave
            result[audio + '_path'] = data_dict[audio]
        return result

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        '''
        Load audio from path, resample it if needed
        :return: 1st channel of audio (tensor of shape 1xL)
        '''
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        '''
        Filter records depending on max_audio_length, max_text_length and limit
        '''
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                    np.array(
                        [len(BaseTextEncoder.normalize_text(el["text"])) for el in index]
                    )
                    >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            # random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )
