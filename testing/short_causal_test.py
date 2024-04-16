import os
import logging 
import warnings

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import wandb

import ss.loss as module_loss
import ss.metric as module_metric
import ss.model as module_arch
from ss.utils import init_obj, MetricTracker
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
    os.environ['MASTER_PORT'] = '12340'
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


def normalize_audio(audio):
    factor = torch.max(torch.abs(audio)).item()
    if factor == 0:
        factor = 1
    audio = audio / factor
    return audio


def run_testing(rank, world_size, config):
    logger = get_logger(config)

    wandb.login()
    if config.get('wandb_project') is None:
        raise ValueError("Please specify project name for wandb")
    wandb.init(
        project=config['wandb_project'],
        name=config['name'],
        config=dict(config)
    )

    device = torch.device("cuda:" + str(rank))
    torch.cuda.set_device(device)

    setup(rank, world_size)
    
    dataloader = get_dataloader(**config["dataset"]["test"])

    streamer = Streamer(**config["streamer"])

    model = init_obj(config["arch"], module_arch, n_speakers=config["n_speakers"])
    model.to(device)
    model = DistributedDataParallel(model)
    logger.info(model)

    if config.get('resume') is None:
        raise ValueError("Please specify resume path")
    logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], device)
    if checkpoint["config"]["arch"] != config["arch"]:
        logger.warning(
            "Warning: Architecture configuration given in config file is different from that "
            "of checkpoint. This may yield an exception while state_dict is being loaded."
        )
    model.load_state_dict(checkpoint["model"])
    logger.info(
        "Checkpoint loaded"
    )

    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}
    metric_tracker = MetricTracker("loss", "SISDR", *sorted(list(metrics.keys())), device=device)
    
    log_step = config["log_step"]
    sr = config["sr"]
    df = pd.DataFrame(columns=["mix", "ref", "target", "pred"])
    df_idx = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="testing", total=len(dataloader))):
            batch["mix_chunks"], n_chunks = streamer.make_chunks(batch["mix"])
            for tensor_for_gpu in ["mix_chunks", "ref", "target", "lens"]:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
            
            batch["s1"], batch["logits"] = model(batch["mix_chunks"], batch["ref"])
            length = batch["target"].shape[-1]
            batch["s1"] = streamer.apply_overlap_add_method(batch["s1"], n_chunks)
            batch["s1"] = batch["s1"][:, :length]
            
            batch["loss"], batch["sisdr"] = criterion(**batch, is_train=False)

            metric_tracker.update("loss", batch["loss"].item())
            metric_tracker.update("SISDR", batch["sisdr"].item())

            for met in metrics.keys():
                metric_tracker.update(met, metrics[met](**batch).item())

            if batch_idx % log_step == 0:
                df.loc[df_idx] = [
                    wandb.Audio(normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=sr),
                    wandb.Audio(normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=sr),
                    wandb.Audio(normalize_audio(batch["target"][0]).detach().cpu().numpy().T, sample_rate=sr),
                    wandb.Audio(normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=sr) 
                ]
                df_idx += 1
    
    wandb.log({"test_results": wandb.Table(dataframe=df)})

    df_vals = pd.DataFrame(columns=["loss", "SISDR", *sorted(list(metrics.keys()))])
    vals = []
    for metric_name in ["loss", "SISDR", *sorted(list(metrics.keys()))]:
        metric_value = metric_tracker.avg(metric_name)
        logger.info("{}: {:.6f}".format(metric_name, metric_value))
        vals.append(metric_value)
    df_vals.loc[0] = vals
    wandb.log({"test_values": wandb.Table(dataframe=df_vals)})
    logger.info('Test results and values are added to wandb')

    cleanup()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    with open_dict(config):
        config.job_logging_config = HydraConfig.get().job_logging 
        config.job_logging_config.handlers.file.filename = HydraConfig.get().runtime.output_dir + '/' + \
                                                                    'ddp_test.log'

    n_gpus = torch.cuda.device_count()
    assert n_gpus == 1, "Require exactly 1 GPU"
    world_size = n_gpus
    mp.spawn(run_testing, 
             args=(world_size, config),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    main()

