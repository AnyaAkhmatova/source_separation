import os
import glob
from glob import glob

from torch.utils.data import Dataset
import torchaudio


class SourceSeparationDataset(Dataset):
    def __init__(self, root, part, max_length=20000, n_speakers=None, filenames_path=None):
        self.path = os.path.join(root, part)
        self.max_length = max_length

        self.file_names = sorted(glob(os.path.join(self.path, '*-mixed.wav')))[:self.max_length]
        
        if filenames_path is None:
            self.target_id_data_id = sorted(list(set([file_name.split('/')[-1].split('_')[0] for file_name in self.file_names])))
            if n_speakers is not None:
                self.target_id_data_id = self.target_id_data_id[: n_speakers]
        else:
            with open(filenames_path, "r") as f:
                lines = f.readlines()
            self.target_id_data_id = []
            for line in lines:
                _, data_id = line.split()
                self.target_id_data_id.append(data_id)
        self.data_id_target_id = {self.target_id_data_id[target_id]: target_id for target_id in range(len(self.target_id_data_id))}
        self.n_speakers = len(self.target_id_data_id)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        mix_name = self.file_names[idx]
        target_name = mix_name.split('-')[0] + '-target.wav'
        ref_name = mix_name.split('-')[0] + '-ref.wav'
        mix, _ = torchaudio.load(mix_name)
        target, _ = torchaudio.load(target_name)
        ref, _ = torchaudio.load(ref_name)
        target_id = self.data_id_target_id.get(mix_name.split('/')[-1].split('_')[0], -100)
        return {
            "mix": mix,
            "target": target,
            "ref": ref,
            "target_id": target_id
        }
