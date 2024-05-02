from abc import abstractmethod
from pathlib import Path

from numpy import inf

import torch
import torch.distributed as dist

from ss.wandb import WanDBWriter


class BaseTrainer:
    def __init__(self, 
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
                 device):
        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.config = config
        if self.rank == 0:
            self.checkpoint_dir = Path(save_dir)
            self.logger = logger
            self.writer = WanDBWriter(
                config, self.logger
            )
        self.device = device
        
        self.last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        if config.get("resume") is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            if self.rank == 0:
                self.logger.info("Saving model on keyboard interrupt")
                self._save_checkpoint(self.last_epoch, save_best=False)
            raise e

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.last_epoch = epoch
            result = self._train_epoch(epoch)

            result_keys = list(result.keys())
            result_tensor = []
            for key in result_keys:
                result_tensor.append(float(result[key]))
            
            result_tensor = torch.tensor(result_tensor, dtype=torch.float32, device=self.device)
            dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
            result_tensor /= self.world_size

            log = {"epoch": epoch}
            for i, key in enumerate(result_keys):
                log[key] = result_tensor[i].item()

            if self.rank == 0:
                for key, value in log.items():
                    self.logger.info("    {:15s}: {}".format(str(key), value))

            best = False
            if self.mnt_mode != "off":
                try:
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    if self.rank == 0:
                        self.logger.warning(
                            "Warning: Metric '{}' is not found. "
                            "Model performance monitoring is disabled.".format(
                                self.mnt_metric
                            )
                        )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count >= self.early_stop:
                    if self.rank == 0:
                        self.logger.info(
                            "Validation performance didn't improve for {} epochs. "
                            "Training stops.".format(self.early_stop)
                        )
                    break
            
                self.lr_scheduler.step(log[self.mnt_metric])

            if (epoch % self.save_period == 0 or best) and self.rank == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "monitor_best": self.mnt_best,
            "config": self.config
        }

        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not save_best:
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        if self.rank == 0:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)

        if checkpoint["config"]["arch"] != self.config["arch"] and self.rank == 0:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["model"], strict=False)

        if self.config.continue_from_checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            self.mnt_best = checkpoint["monitor_best"]

            if ((checkpoint["config"]["optimizer"] != self.config["optimizer"] or
                 checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]) and self.rank == 0):
                self.logger.warning(
                    "Warning: Optimizer or lr_scheduler given in config file is different "
                    "from that of checkpoint. Optimizer and lr_scheduler parameters not being resumed."
                )
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if self.rank == 0:
            self.logger.info(
                "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
            )
