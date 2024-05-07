from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler

from .dataset import SourceSeparationDataset, SourceSeparationInferenceDataset
from .collate_fn import collate_fn, inference_collate_fn


def get_dataloader(root, part, batch_size, max_length=20000, n_speakers=None, filenames_path=None, num_workers=8, pin_memory=True):
    dataset = SourceSeparationDataset(root, part, max_length, n_speakers, filenames_path)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=sampler,
                            num_workers=num_workers, 
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
    return dataloader, sampler


def get_inference_dataloader(root, part, max_length=20000, test_mode=False, num_workers=8):
    dataset = SourceSeparationInferenceDataset(root, part, max_length, test_mode)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            sampler=sampler,
                            num_workers=num_workers, 
                            collate_fn=inference_collate_fn,
                            pin_memory=False)
    return dataloader

