import os
import logging 
import warnings

import numpy as np
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import ss.loss as module_loss
import ss.metric as module_metric
import ss.model as module_arch
from ss.inferencer import CausalInferencer
from ss.utils import init_obj
from ss.datasets import get_inference_dataloader
from ss.streamer import Streamer

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def get_logger(config, name=None):
    logging.config.dictConfig(
        OmegaConf.to_container(config.job_logging_config)
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def run_inferencing(rank, world_size, config):
    if rank == 0:
        logger = get_logger(config)
    else:
        logger = None

    device = torch.device("cuda:" + str(rank))
    torch.cuda.set_device(device)

    setup(rank, world_size)

    model = init_obj(config["arch"], module_arch)
    model.to(device)
    model = DistributedDataParallel(model, find_unused_parameters=True)
    if rank == 0:
        logger.info(model)

    if rank == 0 and config.get('resume') is None:
        raise ValueError("Please specify resume path")
    if rank == 0:
        logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], device)
    if rank == 0 and checkpoint["config"]["arch"] != config["arch"]:
        logger.warning(
            "Warning: Architecture configuration given in config file is different from that "
            "of checkpoint. This may yield an exception while state_dict is being loaded."
        )
    model.load_state_dict(checkpoint["model"])
    if rank == 0:
        logger.info(
            "Checkpoint loaded"
        )

    dataloader = get_inference_dataloader(**config["dataset"]["inference"])
    streamer = Streamer(**config["streamer"])

    test_mode = config["dataset"]["inference"]["test_mode"]
    if test_mode:
        criterion = init_obj(config["loss"], module_loss).to(device)
        metrics = [
            init_obj(metric_dict, module_metric, device=device)
            for metric_dict in config["metrics"]
        ]
        metrics = {met.name: met for met in metrics}
    else:
        criterion = None
        metrics = None
    
    save_inference = config["save_inference"]
    save_dir = config.get("save_dir", None)

    inferencer = CausalInferencer(rank, 
                                 world_size,
                                 model,
                                 dataloader,
                                 streamer,
                                 config,
                                 device,
                                 logger,
                                 test_mode,
                                 criterion,
                                 metrics,
                                 save_inference,
                                 save_dir)

    inferencer.run()

    cleanup()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    with open_dict(config):
        config.job_logging_config = HydraConfig.get().job_logging 
        config.job_logging_config.handlers.file.filename = HydraConfig.get().runtime.output_dir + '/' + \
                                                                    'ddp_inference.log'

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= config["n_gpu"], "Require >= n_gpu GPUs"
    world_size = config["n_gpu"]
    mp.spawn(run_inferencing, 
             args=(world_size, config),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
