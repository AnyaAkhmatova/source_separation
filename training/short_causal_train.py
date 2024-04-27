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
import ss.trainer as module_trainer
from ss.utils import init_obj
from ss.datasets import get_dataloader
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


def run_training(rank, world_size, config, save_dir):
    if rank == 0:
        logger = get_logger(config)
    else:
        logger = None

    device = torch.device("cuda:" + str(rank))
    torch.cuda.set_device(device)

    setup(rank, world_size)

    dataloaders = {}
    samplers = {}
    dataloaders["train"], samplers["train"] = get_dataloader(**config["dataset"]["train"])
    dataloaders["dev"], samplers["dev"] = get_dataloader(**config["dataset"]["dev"])

    streamer = Streamer(**config["streamer"])

    model = init_obj(config["arch"], module_arch)
    model.to(device)
    model = DistributedDataParallel(model)
    if rank == 0:
        logger.info(model)
        logger.info("n_speakers: " + str(dataloaders["train"].dataset.n_speakers))
        logger.info("world_size: " + str(world_size))

    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = init_obj(config["trainer"], 
                       module_trainer, 
                       rank, 
                       world_size,
                       model,
                       criterion,
                       metrics,
                       optimizer,
                       lr_scheduler,
                       config,
                       save_dir,
                       logger,
                       device,
                       dataloaders,
                       samplers,
                       streamer,
                       len_epoch=config["trainer"].get("len_epoch", None))

    trainer.train()

    cleanup()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    with open_dict(config):
        config.job_logging_config = HydraConfig.get().job_logging 
        config.job_logging_config.handlers.file.filename = HydraConfig.get().runtime.output_dir + '/' + \
                                                                    'ddp_train.log'

    save_dir = HydraConfig.get().runtime.output_dir

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, "Require >= 1 GPUs"
    world_size = n_gpus
    mp.spawn(run_training, 
             args=(world_size, config, save_dir),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
