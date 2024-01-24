import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer
from ss.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            save_dir,
            logger,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device, save_dir, logger, lr_scheduler)
        self.skip_oom = skip_oom
        self.config = config

        self.trainer_batch_size = self.config["trainer"]["trainer_batch_size"]
        self.dataset_batch_size = self.config["trainer"]["dataset_batch_size"]
        
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            self.len_epoch = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = max(self.len_epoch // 10, 1)

        self.train_metrics_names = self.config["train_metrics"]
        self.train_metrics = MetricTracker(
            "loss", "grad norm", *self.train_metrics_names, writer=self.writer
        )
        self.eval_metrics_names = self.config["dev_metrics"]
        self.evaluation_metrics = MetricTracker(
            "loss", *self.eval_metrics_names, writer=self.writer
        )

        self.mode = "train"

    @staticmethod
    def move_batch_to_device(batch, device, is_train):
        for tensor_for_gpu in ["mix", "ref", "target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        if is_train:
            batch["target_id"] = batch["target_id"].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        self.mode = "train"
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.len_epoch)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f} "
                    "SISDR: {:.4f} PESQ: {:.4f} ACC: {:.4f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item(),
                        batch["sisdr"], batch["pesq"], batch["acc"]
                    )
                )
                if self.lr_scheduler is not None:
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                self._log_sample(batch)
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, metrics, batch_idx=0):
        is_train = (self.mode == "train")
        metric_names = self.train_metrics_names if is_train else self.eval_metrics_names
        
        batch = self.move_batch_to_device(batch, self.device, is_train)
        if is_train:
            if (batch_idx * self.dataset_batch_size) % self.trainer_batch_size == 0: 
                self.optimizer.zero_grad()
        s1, s2, s3, logits = self.model(batch["mix"], batch["ref"])
        batch["s1"], batch["s2"], batch["s3"], batch["logits"] = s1, s2, s3, logits
        batch["loss"] = self.criterion(**batch, is_train=is_train)
        if is_train:
            batch["loss"].backward()
            if ((batch_idx + 1) * self.dataset_batch_size) % self.trainer_batch_size == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
        metrics.update("loss", batch["loss"].item())

        for met in metric_names:
            batch[met.lower()] = self.metrics[met](**batch)
            metrics.update(met, batch[met.lower()])
        
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.mode = "eval"
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch in tqdm(
                    dataloader,
                    desc=part,
                    total=len(dataloader)
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_sample(batch)

        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx * self.dataset_batch_size
            total = self.len_epoch * self.dataset_batch_size
        return base.format(current, total, 100.0 * current / total)

    def _log_sample(self, batch):
        ind = np.random.choice(batch["mix"].shape[0])
        mix = batch["mix"][ind]
        ref = batch["ref"][ind]
        target = batch["target"][ind]
        sr = self.config["dataset"]["sr"]
        self.writer.add_audio("mix_audio", mix, sr)
        self.writer.add_audio("ref_audio", ref, sr)
        self.writer.add_audio("target_audio", target, sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker):
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
