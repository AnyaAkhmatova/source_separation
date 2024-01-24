import os
import glob
from glob import glob

from torch.utils.data import Dataset
import torchaudio


class SourceSeparationDataset(Dataset):
    def __init__(self, root, part, max_length=20000):
        self.path = os.path.join(root, part)
        self.max_length = max_length
        self.mix = sorted(glob(os.path.join(self.path, '*-mixed.wav')))[:max_length]
        self.target = sorted(glob(os.path.join(self.path, '*-target.wav')))[:max_length]
        self.ref = sorted(glob(os.path.join(self.path, '*-ref.wav')))[:max_length]
        self.target_id_data_id = sorted(list(set([mix.split('/')[-1].split('_')[0] for mix in self.mix])))
        self.data_id_target_id = {self.target_id_data_id[target_id]: target_id for target_id in range(len(self.target_id_data_id))}
        self.n_speakers = len(self.target_id_data_id)

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        mix, _ = torchaudio.load(self.mix[idx])
        target, _ = torchaudio.load(self.target[idx])
        ref, _ = torchaudio.load(self.ref[idx])
        target_id = self.data_id_target_id[self.mix[idx].split('/')[-1].split('_')[0]]
        return {
            "mix": mix,
            "target": target,
            "ref": ref,
            "target_id": target_id
        }
