import logging 
import warnings

import hydra

import numpy as np
import torch

import ss.loss as module_loss
import ss.metric as module_metric
import ss.model as module_arch
from ss.trainer import Trainer
from ss.utils import prepare_device, init_obj
from ss.datasets import get_dataloader

warnings.filterwarnings("ignore")


SEED = 105
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    save_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir    

    dataloaders = {}
    dataloaders["train"] = get_dataloader(**config["dataset"]["train"])
    dataloaders["dev"] = get_dataloader(**config["dataset"]["dev"])

    model = init_obj(config["arch"], module_arch, n_speakers=dataloaders["train"].dataset.n_speakers)
    logger.info(model)
    logger.info("n_speakers: " + str(dataloaders["train"].dataset.n_speakers))

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(device)

    criterion = init_obj(config["loss"], module_loss).to(device)
    metrics = [
        init_obj(metric_dict, module_metric, device=device)
        for metric_dict in config["metrics"]
    ]
    metrics = {met.name: met for met in metrics}

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = None
    if config.get("lr_scheduler") is not None:
        lr_scheduler = init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        save_dir,
        logger,
        dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    main()
