from torch.utils.data import DataLoader 
from torch.utils.data.distributed import DistributedSampler

from .dataset import SourceSeparationDataset
from .collate_fn import collate_fn


def get_dataloader(root, part, batch_size, max_length=20000, num_workers=8, pin_memory=True):
    dataset = SourceSeparationDataset(root, part, max_length)
    shuffle = False
    if part == 'train':
        shuffle = True
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=DistributedSampler(dataset, shuffle=shuffle),
                            num_workers=num_workers, 
                            collate_fn=collate_fn,
                            pin_memory=pin_memory)
    return dataloader
