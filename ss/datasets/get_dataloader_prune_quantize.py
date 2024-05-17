from torch.utils.data import DataLoader 

from .dataset import SourceSeparationDataset, SourceSeparationInferenceDataset
from .collate_fn import collate_fn, inference_collate_fn


def get_simple_dataloader(root, part, batch_size, max_length=20000, n_speakers=None, filenames_path=None, num_workers=8, pin_memory=True):
    dataset = SourceSeparationDataset(root, part, max_length, n_speakers, filenames_path)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=num_workers, 
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
    return dataloader


def get_simple_inference_dataloader(root, part, max_length=20000, test_mode=False, num_workers=8):
    dataset = SourceSeparationInferenceDataset(root, part, max_length, test_mode)
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=num_workers, 
                            collate_fn=inference_collate_fn,
                            pin_memory=False)
    return dataloader

