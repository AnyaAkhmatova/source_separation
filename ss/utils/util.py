from itertools import repeat
import importlib

import pandas as pd
import torch


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def init_obj(obj_dict, default_module, *args, **kwargs):
    if "module" in obj_dict:
        default_module = importlib.import_module(obj_dict["module"])
    module_name = obj_dict["type"]
    module_args = dict(obj_dict["args"])
    assert all(
        [k not in module_args for k in kwargs]
    ), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    return getattr(default_module, module_name)(*args, **module_args)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
