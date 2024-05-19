import gc
from time import time
import logging 
import warnings

import numpy as np
import random
from tqdm import tqdm

import torch

from thop import profile

import hydra

import ss.model as module_arch
from ss.utils import init_obj
from ss.datasets import get_simple_dataloader
from ss.streamer import Streamer

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def start_timer(gpu):
    if gpu:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    start_time = time()
    return start_time


def end_timer(gpu):
    if gpu:
        torch.cuda.synchronize()
    end_time = time()
    if gpu:
        memory_used = torch.cuda.max_memory_allocated() // 2**20
        return end_time, memory_used
    return end_time


def run(config, logger, device):    
    
    dataloader = get_simple_dataloader(**config["dataset"]["test"])
    assert config["dataset"]["test"]["batch_size"] == 1, "batch_size must be 1"

    streamer = Streamer(**config["streamer"])

    gpu = (config["mode"] == "gpu")

    model = init_obj(config["arch"], module_arch)
    model.to(device)
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
    model_state_dict = {}
    for key, value in checkpoint["model"].items():
        model_state_dict['.'.join(key.split('.')[1:])] = value
    logger.info(model.load_state_dict(model_state_dict))
    logger.info(
        "Checkpoint loaded"
    )

    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of model's parameters: " + str(model_total_params))

    model = model.to("cpu")
    batch = dataloader.dataset[0]
    mix, ref = batch["mix"][:, :config["dataset"]["sr"]*1], batch["ref"][:, :config["dataset"]["sr"]*2]
    mix_chunks, n_chunks = streamer.make_chunks(mix)
    macs, params = profile(model, inputs=(mix_chunks, ref, False))
    logger.info(f"macs and params: {macs/1e9}, {params}")
    model = model.to(device)

    num_iters = config["num_iters"]
    times = []
    if gpu:
        memories = []
    mix_lengths = []
    ref_lengths = []
    chunks_nums = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="measuring_gpu_time", total=num_iters)):
            if batch_idx >= num_iters:
                break

            mix, ref = batch["mix"], batch["ref"]
            mix_chunks, n_chunks = streamer.make_chunks(mix)
            mix_chunks = mix_chunks.to(device)
            ref = ref.to(device)

            start = start_timer(gpu)
            s1 = model(mix_chunks, ref, False)
            results = end_timer(gpu)
            if gpu:
                end, memory = results
            else:
                end = results

            times.append(end - start)
            if gpu:
                memories.append(memory)
            mix_lengths.append(mix.shape[-1]/config["dataset"]["sr"])
            ref_lengths.append(ref.shape[-1]/config["dataset"]["sr"])
            chunks_nums.append(n_chunks)

    mean_time = np.array(times).mean()
    if gpu:
        mean_memory = np.array(memories).mean()
    mean_mix_length = np.array(mix_lengths).mean()
    mean_ref_length = np.array(ref_lengths).mean()
    mean_chunks_num = np.array(chunks_nums).mean()
    if gpu:
        logger.info(f"Average gpu time per sample (s): {mean_time}")
        logger.info(f"Average gpu memory used per sample (Mb): {mean_memory}")
    else:
        logger.info(f"Average cpu time per sample (s): {mean_time}")
    logger.info(f"Average mix length (s): {mean_mix_length}")
    logger.info(f"Average ref length (s): {mean_ref_length}")
    logger.info(f"Average number of chunks: {mean_chunks_num}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if config["mode"] == "gpu":
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 1, "Require >= 1 GPU"
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    run(config, logger, device)
    

if __name__ == "__main__":
    main()

