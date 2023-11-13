import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    assert len(dataset_items) > 0

    result_batch = {}

    mix_lengths, ref_lengths = [], []
    for item in dataset_items:
        mix_lengths.append(item['mix'].size(1))
        ref_lengths.append(item['ref'].size(1))
    result_batch['mix_lengths'] = torch.tensor(mix_lengths)
    result_batch['ref_lengths'] = torch.tensor(ref_lengths)

    max_mix_len, max_ref_len = max(mix_lengths), max(ref_lengths)
    for key, max_len in zip(['mix', 'ref', 'target'], [max_mix_len, max_ref_len, max_mix_len]):
        result_batch[key] = torch.zeros(len(dataset_items), max_len)
        for idx, item in enumerate(dataset_items):
            result_batch[key][idx, :item[key].size(1)] = item[key].squeeze(0)

    for key in ['mix_path', 'ref_path', 'target_path']:
        result_batch[key] = [item[key] for item in dataset_items]

    if 'speaker_id' in dataset_items[0]:
        result_batch['speaker_id'] = torch.tensor([item['speaker_id'] for item in dataset_items])

    return result_batch
