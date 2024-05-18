import os
import logging 
import warnings

import numpy as np
import random

import torch
from torch import nn
from torch.nn import DataParallel

from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter, get_prunable_layers, tensor_sparsity

import onnxruntime

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

    speaker_handler.eval()
    main_model.train()
    
    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}

    additional_steps = 0

    if config["prune"]:
        dataloaders = {}
        dataloaders["train"] = get_simple_dataloader(**config["dataset"]["train"])
        dataloaders["dev"] = get_simple_dataloader(**config["dataset"]["dev"])

        trainable_params = filter(lambda p: p.requires_grad, main_model.parameters())
        optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)

        manager = ScheduledModifierManager.from_yaml(config["prune_recipe_path"])
        optimizer = manager.modify(main_model, optimizer, 
                        steps_per_epoch=(
                            len(dataloaders["train"]) // \
                            (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"])
                        )
                    )
        config["trainer"]["epochs"] = config["prune_epochs"]

        trainer = SimpleShortCausalTrainer(speaker_handler,
                                           main_model,
                                           criterion,
                                           metrics,
                                           optimizer,
                                           config,
                                           logger,
                                           device,
                                           dataloaders,
                                           streamer,
                                           len_epoch=config["trainer"].get("len_epoch", None))
        trainer.train()

        manager.finalize(main_model)

        additional_steps = config["trainer"]["epochs"] * (len(dataloaders["train"]) // \
                            (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"]))

    if config["quantize"]:
        config["trainer"]["epochs"] = config["quantize_epochs"]
        config["dataset"]["train"]["max_length"] = int(config["dataset"]["train"]["max_length"] * 0.1)
        config["dataset"]["dev"]["max_length"] = int(config["dataset"]["dev"]["max_length"] * 0.1)

        dataloaders = {}
        dataloaders["train"] = get_simple_dataloader(**config["dataset"]["train"])
        dataloaders["dev"] = get_simple_dataloader(**config["dataset"]["dev"])

        trainable_params = filter(lambda p: p.requires_grad, main_model.parameters())
        optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)

        manager = ScheduledModifierManager.from_yaml(config["quntize_recipe_path"])
        optimizer = manager.modify(main_model, optimizer, 
                        steps_per_epoch=(
                            len(dataloaders["train"]) // \
                            (config["trainer"]["batch_size"] // config["dataset"]["train"]["batch_size"])
                        )
                    )

        trainer = SimpleShortCausalTrainer(speaker_handler,
                                           main_model,
                                           criterion,
                                           metrics,
                                           optimizer,
                                           config,
                                           logger,
                                           device,
                                           dataloaders,
                                           streamer,
                                           len_epoch=config["trainer"].get("len_epoch", None), 
                                           additional_steps=additional_steps)
    
        trainer.train()

        manager.finalize(main_model)

    if config["prune"]:    
        logger.info("main_model sparsity:")
        for (name, layer) in get_prunable_layers(main_model):
            logger.info(f"{name}.weight: {tensor_sparsity(layer.weight).item():.4f}")

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
