import importlib

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist


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
    def __init__(self, *keys, device=None):
        self.device = device
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += float(value * n)
        self._data.counts[key] += float(n)
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
    
    def result_sync(self):
        tensor = torch.tensor(self._data.values.astype(np.float32), dtype=torch.float32, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        temp = pd.DataFrame(data=tensor.cpu().numpy(), index=self._data.index, columns=self._data.columns)
        for key in temp.total.keys():
            temp.average[key] = temp.total[key] / temp.counts[key]
        return dict(temp.average)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
