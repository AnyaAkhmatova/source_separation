import os
import logging 
import warnings

import numpy as np
import random

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, get_prunable_layers, tensor_sparsity

import onnxruntime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

import ss.loss as module_loss
import ss.metric as module_metric
from ss.model import SpexPlusShortSpeakerHandler, SpexPlusShortGRUMainModel
from ss.trainer import SimpleShortCausalTrainer
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

    speaker_handler = SpexPlusShortSpeakerHandler(**config["speaker_handler"])
    speaker_handler = speaker_handler.to(device)
    speaker_handler = DistributedDataParallel(speaker_handler, find_unused_parameters=True)

    main_model = SpexPlusShortGRUMainModel(**config["main_model"])
    main_model = main_model.to(device)
    main_model = DistributedDataParallel(main_model, find_unused_parameters=True)

    if rank == 0:
        logger.info(speaker_handler)
        logger.info(main_model)
        logger.info("n_speakers: " + str(config["dataset"]["train"]["n_speakers"]))

    if rank == 0 and config.get('resume') is None:
        raise ValueError("Please specify resume path")
    if rank == 0:
        logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], device)
    speaker_handler.load_state_dict(checkpoint["model"], strict=False)
    main_model.load_state_dict(checkpoint["model"], strict=False)
    if rank == 0:
        logger.info(
            "Checkpoint loaded"
        )

    speaker_handler.eval()
    main_model.train()
    
    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}

    trainable_params = filter(lambda p: p.requires_grad, main_model.parameters())
    optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)

    manager = ScheduledModifierManager.from_yaml(config["recipe_path"])
    optimizer = manager.modify(main_model, optimizer, 
                    steps_per_epoch=(
                        len(dataloaders["train"]) // \
                        (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"])
                    )
                )

    trainer = SimpleShortCausalTrainer(rank, 
                                       world_size, 
                                       speaker_handler,
                                       main_model,
                                       criterion,
                                       metrics,
                                       optimizer,
                                       config,
                                       logger,
                                       device,
                                       dataloaders, 
                                       samplers,
                                       streamer,
                                       len_epoch=config["trainer"].get("len_epoch", None))
    trainer.train()

    manager.finalize(main_model)

    if rank == 0 and config["prune"]:    
        logger.info("main_model sparsity:")
        for (name, layer) in get_prunable_layers(main_model):
            logger.info(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")
    
    if rank == 0:
        speaker_handler = speaker_handler.module.to("cpu")
        main_model = main_model.module.to("cpu")

        speaker_handler.eval()
        main_model.eval()

        logger.info("export speaker handler")
        exporter1 = ModuleExporter(speaker_handler, output_dir=save_dir)
        exporter1.export_pytorch(name=config["speaker_handler_name"]+".pth")
        exporter1.export_onnx((torch.randn(1, 160000), ), 
                            name=config["speaker_handler_name"]+".onnx", 
                            convert_qat=False,
                            input_names=["ref"], 
                            output_names=["ref_vec", "speaker_logits"], 
                            dynamic_axes={"ref": {1: "ref_length"}})
        
        logger.info("export main model")
        exporter2 = ModuleExporter(main_model, output_dir=save_dir)
        exporter2.export_pytorch(name=config["main_model_name"]+".pth")
        exporter2.export_onnx((torch.randn(1, config["streamer"]["chunk_window"]), 
                            torch.randn(1, config["main_model"]["out_channels"]), 
                            torch.randn(1, config["main_model"]["memory_size"], config["time_dim"])), 
                            name="sparse_"+config["main_model_name"]+".onnx", 
                            convert_qat=config["quantize"],
                            input_names=["chunk", "ref_vec", "memory"], 
                            output_names=["s1_chunk", "new_memory"])
        
        batch = dataloaders["dev"].dataset[0]

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        model = speaker_handler
        model_input = (batch["ref"], )
        model_ref_vec, model_logits = model(*model_input)

        ort_session = onnxruntime.InferenceSession(os.path.join(save_dir, config["speaker_handler_name"]+".onnx"), 
                                                providers=['CPUExecutionProvider'])
        onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), model_input)}
        onnxruntime_ref_vec, onnxruntime_logits = ort_session.run(None, onnxruntime_input)

        difference = [torch.max(torch.abs(model_ref_vec - torch.tensor(onnxruntime_ref_vec))).item(), 
                    torch.max(torch.abs(model_logits - torch.tensor(onnxruntime_logits))).item()]

        logger.info(f"speaker handler difference: {difference}")

        model = main_model
        model_input = (batch["mix"][:, : config["streamer"]["chunk_window"]], 
                    model_ref_vec, 
                    torch.zeros((1, config["main_model"]["memory_size"], config["time_dim"])))
        model_s1, model_memory = model(*model_input)

        ort_session = onnxruntime.InferenceSession(os.path.join(save_dir, "sparse_"+config["main_model_name"]+".onnx"), 
                                                providers=['CPUExecutionProvider'])
        onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), model_input)}
        onnxruntime_s1, onnxruntime_memory = ort_session.run(None, onnxruntime_input)

        difference = [torch.max(torch.abs(model_s1 - torch.tensor(onnxruntime_s1))).item(), 
                    torch.max(torch.abs(model_memory - torch.tensor(onnxruntime_memory))).item()]

        logger.info(f"main model difference: {difference}")

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
