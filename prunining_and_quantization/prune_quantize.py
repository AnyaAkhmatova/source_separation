import logging 
import warnings

import numpy as np
import random

import torch
from torch.nn import DataParallel

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, get_prunable_layers, tensor_sparsity

import hydra
from hydra.core.hydra_config import HydraConfig

import ss.loss as module_loss
import ss.metric as module_metric
from ss.model import SpexPlusShortSpeakerHandler, SpexPlusShortGRUMainModel
from ss.trainer import SimpleShortCausalTrainer
from ss.utils import init_obj, prepare_device
from ss.datasets import get_simple_dataloader
from ss.streamer import Streamer

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def run_training(config, save_dir, logger):
    device, list_ids = prepare_device(config["n_gpu"])

    dataloaders = {}
    dataloaders["train"] = get_simple_dataloader(**config["dataset"]["train"])
    dataloaders["dev"] = get_simple_dataloader(**config["dataset"]["dev"])

    streamer = Streamer(**config["streamer"])

    speaker_handler = SpexPlusShortSpeakerHandler(**config["speaker_handler"])
    speaker_handler = speaker_handler.to(device)
    speaker_handler = DataParallel(speaker_handler, device_ids=list_ids)
    main_model = SpexPlusShortGRUMainModel(**config["main_model"])
    main_model = main_model.to(device)
    main_model = DataParallel(main_model, device_ids=list_ids)

    logger.info(speaker_handler)
    logger.info(main_model)
    logger.info("n_speakers: " + str(dataloaders["train"].dataset.n_speakers))

    if config.get('resume') is None:
        raise ValueError("Please specify resume path")
    logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], device)
    speaker_handler.load_state_dict(checkpoint["model"], strict=False)
    main_model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(
        "Checkpoint loaded"
    )
    
    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}

    trainable_params1 = filter(lambda p: p.requires_grad, speaker_handler.parameters())
    optimizer1 = init_obj(config["optimizer"], torch.optim, trainable_params1)
    trainable_params2 = filter(lambda p: p.requires_grad, main_model.parameters())
    optimizer2 = init_obj(config["optimizer"], torch.optim, trainable_params2)

    manager1 = ScheduledModifierManager.from_yaml(config["recipe_path"])
    optimizer1 = manager1.modify(speaker_handler, optimizer1, 
                    steps_per_epoch=(
                        len(dataloaders["train"]) // \
                        (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"])
                    )
                )
    manager2 = ScheduledModifierManager.from_yaml(config["recipe_path"])
    optimizer2 = manager2.modify(main_model, optimizer2, 
                    steps_per_epoch=(
                        len(dataloaders["train"]) // \
                        (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"])
                    )
                )

    trainer = SimpleShortCausalTrainer(speaker_handler,
                                       main_model,
                                       criterion,
                                       metrics,
                                       optimizer1,
                                       optimizer2,
                                       config,
                                       logger,
                                       device,
                                       dataloaders,
                                       streamer,
                                       len_epoch=config["trainer"].get("len_epoch", None))

    trainer.train()

    manager1.finalize(speaker_handler)
    manager2.finalize(main_model)
    logger.info("speaker_handler sparsity:")
    for (name, layer) in get_prunable_layers(speaker_handler):
        logger.info(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")
    logger.info("main_model sparsity:")
    for (name, layer) in get_prunable_layers(main_model):
        logger.info(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")

    speaker_handler.eval()
    main_model.eval()
    exporter1 = ModuleExporter(speaker_handler, output_dir=save_dir)
    exporter1.export_pytorch(name=config["speaker_handler_name"]+".pth")
    exporter1.export_onnx((torch.randn(1, 160000),), 
                          name="sparse_"+config["speaker_handler_name"]+".onnx", 
                          convert_qat=True,
                          input_names=["ref"], 
                          output_names=["ref_vec", "speaker_logits"], 
                          dynamic_axes={"ref": [1]})
    exporter2 = ModuleExporter(main_model, output_dir=save_dir)
    exporter2.export_pytorch(name=config["main_model_name"]+".pth")
    exporter2.export_onnx((torch.randn(1, config["streamer"]["chunk_window"]), 
                           torch.randn(1, config["main_model"]["out_channels"]), 
                           torch.randn(1, config["main_model"]["memory_size"], config["streamer"]["chunk_window"])), 
                          name="sparse_"+config["main_model_name"]+".onnx", 
                          convert_qat=True,
                          input_names=["chunk", "ref_vec", "memory"], 
                          output_names=["s1_chunk", "new_memory"])


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    save_dir = HydraConfig.get().runtime.output_dir
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= config["n_gpu"], "Require >= n_gpu GPUs"
    
    run_training(config, save_dir, logger)


if __name__ == "__main__":
    main()
