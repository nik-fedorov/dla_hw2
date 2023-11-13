from glob import glob
import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from ss.base.base_mixture_dataset import BaseMixtureDataset
from ss.utils import ROOT_PATH
from ss.utils.mixture import MixtureGenerator

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id, audios_dir, audioTemplate="*-norm.wav"):
        self.id = speaker_id
        self.files = []
        self.audioTemplate=audioTemplate
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        speakerDir = os.path.join(audios_dir,self.id) #it is a string
        chapterDirs = os.scandir(speakerDir)
        files=[]
        for chapterDir in chapterDirs:
            files = files + [file for file in glob(os.path.join(speakerDir,chapterDir.name)+"/"+self.audioTemplate)]
        return files


class LibrispeechMixtureDataset(BaseMixtureDataset):
    def __init__(self, part, n_samples, audio_len=3, data_dir=None, mixture_data_dir=None, num_workers=2, *args, **kwargs):
        assert part in URL_LINKS

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        _ = self._get_or_load_index(part)

        # prepare mixture data
        if mixture_data_dir is None:
            mixture_data_dir = ROOT_PATH / "data" / "datasets" / "librispeech_mixture"
            mixture_data_dir.mkdir(exist_ok=True, parents=True)
        self._mixture_data_dir = mixture_data_dir

        path = os.path.join(self._data_dir, part)
        mixture_path = os.path.join(self._mixture_data_dir, f'{part}-{n_samples}')

        self.speakers = [el.name for el in os.scandir(path)]
        if Path(mixture_path).exists():
            logger.info(f'Mixture dataset with path "{mixture_path}" already exists. Use it.')
        else:
            logger.info(f'Generating mixture dataset with path "{mixture_path}" ...')
            speakers_files = [LibriSpeechSpeakerFiles(i, path, audioTemplate="*.flac") for i in self.speakers]
            mixer = MixtureGenerator(speakers_files, mixture_path, nfiles=n_samples,
                                     test='test' in part or 'dev' in part)
            mixer.generate_mixes(snr_levels=[-5, 5], num_workers=num_workers, update_steps=100,
                                 trim_db=20, vad_db=20, audioLen=audio_len)

        mix = sorted(glob(os.path.join(mixture_path, '*-mixed.wav')))
        ref = sorted(glob(os.path.join(mixture_path, '*-ref.wav')))
        target = sorted(glob(os.path.join(mixture_path, '*-target.wav')))

        index = []
        for mix_path, ref_path, target_path in zip(mix, ref, target):
            entry = {
                'mix': mix_path,
                'ref': ref_path,
                'target': target_path,
            }
            if 'train' in part:
                speaker_id = self.speakers.index(ref_path.split('/')[-1].split('_')[0])
                entry['speaker_id'] = speaker_id
            index.append(entry)

        super().__init__(index, *args, **kwargs)

    @property
    def num_speakers(self):
        return len(self.speakers)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
