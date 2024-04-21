import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from .base_trainer import BaseTrainer
from ss.utils import MetricTracker


class Trainer(BaseTrainer):
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
                 device,
                 dataloaders,
                 sampler,
                 len_epoch=None,
                 skip_oom=True):
        super().__init__(rank, 
                         world_size, 
                         model, 
                         criterion, 
                         metrics, 
                         optimizer, 
                         lr_scheduler,
                         config, 
                         save_dir, 
                         logger, 
                         device)
        
        self.skip_oom = skip_oom
        
        self.train_dataloader = dataloaders["train"]        
        self.train_sampler = sampler
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        self.batch_size = self.config["trainer"]["batch_size"]
        self.data_batch_size = self.config["dataset"]["train"]["batch_size"]
        self.grad_accum_step = self.batch_size // self.data_batch_size
        self.sr = self.config["dataset"]["sr"]
        if len_epoch is not None:
            assert len_epoch <= len(self.train_dataloader), "len_epoch should be less or equal to the length of train_dataloader"
        self.len_epoch = len_epoch if len_epoch is not None else len(self.train_dataloader)
        self.log_step = max(self.len_epoch // 5, 1)

        self.train_metrics_names = self.config["train_metrics"]
        self.train_metrics = MetricTracker(
            "loss", "SISDR", "grad norm", *self.train_metrics_names, device=self.device
        )
        self.eval_metrics_names = self.config["dev_metrics"]
        self.evaluation_metrics = MetricTracker(
            "loss", "SISDR", *self.eval_metrics_names, device=self.device
        )

        self.mode = "train"
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_epoch(self, epoch):
        self.mode = "train"
        self.model.train()
        self.train_metrics.reset()
        self.train_sampler.set_epoch(epoch)

        if self.rank == 0:
            self.writer.set_step((epoch - 1) * self.len_epoch)
            self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch) if self.rank == 0 else self.train_dataloader
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    metric_names=self.train_metrics_names,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    if self.rank == 0:
                        self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm().item())

            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch - 1:
                log_tensor = torch.tensor([batch["loss"], batch["sisdr"], batch["acc"]], dtype=torch.float32, device=self.device)
                dist.all_reduce(log_tensor, op=dist.ReduceOp.SUM)
                log_tensor /= self.world_size

                if self.rank == 0:
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f} SISDR: {:.4f} ACC: {:.4f}".format(
                            epoch, self._progress(batch_idx), 
                            log_tensor[0].item(), log_tensor[1].item(), log_tensor[2].item()
                        )
                    )
                
                cur_result = self.train_metrics.result_sync()

                if self.rank == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.state_dict()['param_groups'][0]['lr']
                    )
                    self._log_scalars(cur_result)
                    self._log_sample(batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
    
    def _evaluation_epoch(self, epoch, part, dataloader):
        self.mode = "eval"
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch in (
                tqdm(dataloader, desc=part, total=len(dataloader)) if self.rank == 0 else dataloader
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                    metric_names=self.eval_metrics_names
                )
            
            cur_result = self.evaluation_metrics.result_sync()

            if self.rank == 0:
                self.writer.set_step(epoch * self.len_epoch, part)
                self._log_scalars(cur_result)
                self._log_sample(batch)

        return self.evaluation_metrics.result()
    
    @staticmethod
    def move_batch_to_device(batch, device, is_train):
        for tensor_for_gpu in ["mix", "ref", "target", "lens"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device, non_blocking=True)
        if is_train:
            batch["target_id"] = batch["target_id"].to(device, non_blocking=True)
        return batch
    
    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def process_batch(self, batch, metrics, metric_names, batch_idx=0):
        is_train = (self.mode == "train")

        if is_train:
            batch = self.move_batch_to_device(batch, self.device, is_train)
            if batch_idx % self.grad_accum_step == 0:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                s1, s2, s3, logits = self.model(batch["mix"], batch["ref"])
                batch["s1"], batch["s2"], batch["s3"], batch["logits"] = s1, s2, s3, logits
                batch["loss"], batch["sisdr"] = self.criterion(**batch, is_train=is_train)
            self.scaler.scale(batch["loss"] / self.grad_accum_step).backward()
            if (batch_idx + 1) % self.grad_accum_step == 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            batch = self.move_batch_to_device(batch, self.device, is_train)
            s1, s2, s3 = self.model(batch["mix"], batch["ref"])
            batch["s1"], batch["s2"], batch["s3"] = s1, s2, s3
            batch["loss"], batch["sisdr"] = self.criterion(**batch, is_train=is_train)
        
        metrics.update("loss", batch["loss"].item())
        metrics.update("SISDR", batch["sisdr"].item())

        for met in metric_names:
            batch[met.lower()] = self.metrics[met](**batch)
            metrics.update(met, batch[met.lower()].item())
        
        return batch
    
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
        return total_norm

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx * self.data_batch_size
        total = self.len_epoch * self.data_batch_size
        return base.format(current, total, 100.0 * current / total)
    
    def _normalize_audio(self, audio):
        factor = torch.max(torch.abs(audio)).item()
        if factor == 0:
            factor = 1
        audio = audio / factor
        return audio
    
    def _log_scalars(self, cur_result):
        for metric_name, metric_value in cur_result.items():
            self.writer.add_scalar(f"{metric_name}", metric_value)

    def _log_sample(self, batch):
        ind = np.random.choice(batch["mix"].shape[0])
        mix_len = int(batch["lens"][ind].item())
        mix = self._normalize_audio(batch["mix"][ind][:mix_len].type(torch.float32))
        ref = self._normalize_audio(batch["ref"][ind].type(torch.float32))
        target = self._normalize_audio(batch["target"][ind][:mix_len].type(torch.float32))
        pred = self._normalize_audio(batch["s1"][ind][:mix_len].type(torch.float32))        
        self.writer.add_audio("mix_audio", mix, self.sr)
        self.writer.add_audio("ref_audio", ref, self.sr)
        self.writer.add_audio("target_audio", target, self.sr)
        self.writer.add_audio("pred_audio", pred, self.sr)


class CausalTrainer(BaseTrainer):
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
                 device,
                 dataloaders,
                 sampler,
                 streamer,
                 len_epoch=None,
                 skip_oom=True):
        super().__init__(rank, 
                         world_size, 
                         model, 
                         criterion, 
                         metrics, 
                         optimizer, 
                         lr_scheduler,
                         config, 
                         save_dir, 
                         logger, 
                         device)
        
        self.skip_oom = skip_oom
        
        self.train_dataloader = dataloaders["train"]        
        self.train_sampler = sampler
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.streamer = streamer

        self.batch_size = self.config["trainer"]["batch_size"]
        self.data_batch_size = self.config["dataset"]["train"]["batch_size"]
        self.grad_accum_step = self.batch_size // self.data_batch_size
        self.sr = self.config["dataset"]["sr"]
        if len_epoch is not None:
            assert len_epoch <= len(self.train_dataloader), "len_epoch should be less or equal to the length of train_dataloader"
        self.len_epoch = len_epoch if len_epoch is not None else len(self.train_dataloader)
        self.log_step = max(self.len_epoch // 5, 1)

        self.train_metrics_names = self.config["train_metrics"]
        self.train_metrics = MetricTracker(
            "loss", "SISDR", "grad norm", *self.train_metrics_names, device=self.device
        )
        self.eval_metrics_names = self.config["dev_metrics"]
        self.evaluation_metrics = MetricTracker(
            "loss", "SISDR", *self.eval_metrics_names, device=self.device
        )

        self.mode = "train"
        self.scaler = torch.cuda.amp.GradScaler()

    def _train_epoch(self, epoch):
        self.mode = "train"
        self.model.train()
        self.train_metrics.reset()
        self.train_sampler.set_epoch(epoch)

        if self.rank == 0:
            self.writer.set_step((epoch - 1) * self.len_epoch)
            self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch) if self.rank == 0 else self.train_dataloader
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                    metric_names=self.train_metrics_names,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    if self.rank == 0:
                        self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm().item())

            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch - 1:
                log_tensor = torch.tensor([batch["loss"], batch["sisdr"], batch["acc"]], dtype=torch.float32, device=self.device)
                dist.all_reduce(log_tensor, op=dist.ReduceOp.SUM)
                log_tensor /= self.world_size

                if self.rank == 0:
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f} SISDR: {:.4f} ACC: {:.4f}".format(
                            epoch, self._progress(batch_idx), 
                            log_tensor[0].item(), log_tensor[1].item(), log_tensor[2].item()
                        )
                    )
                
                cur_result = self.train_metrics.result_sync()

                if self.rank == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.writer.add_scalar(
                        "learning rate", self.optimizer.state_dict()['param_groups'][0]['lr']
                    )
                    self._log_scalars(cur_result)
                    self._log_sample(batch)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
    
    def _evaluation_epoch(self, epoch, part, dataloader):
        self.mode = "eval"
        self.model.eval()
        self.evaluation_metrics.reset()

        with torch.no_grad():
            for batch in (
                tqdm(dataloader, desc=part, total=len(dataloader)) if self.rank == 0 else dataloader
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                    metric_names=self.eval_metrics_names
                )
            
            cur_result = self.evaluation_metrics.result_sync()

            if self.rank == 0:
                self.writer.set_step(epoch * self.len_epoch, part)
                self._log_scalars(cur_result)
                self._log_sample(batch)

        return self.evaluation_metrics.result()
    
    @staticmethod
    def move_batch_to_device(batch, device, is_train):
        for tensor_for_gpu in ["mix_chunks", "ref", "target", "lens"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device, non_blocking=True)
        if is_train:
            batch["target_id"] = batch["target_id"].to(device, non_blocking=True)
        return batch
    
    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def process_batch(self, batch, metrics, metric_names, batch_idx=0):
        is_train = (self.mode == "train")

        if is_train:
            batch["mix_chunks"], n_chunks = self.streamer.make_chunks(batch["mix"])
            batch = self.move_batch_to_device(batch, self.device, is_train)

            if batch_idx % self.grad_accum_step == 0:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                batch["s1"], batch["s2"], batch["s3"], batch["logits"] = self.model(batch["mix_chunks"], batch["ref"])
                length = batch["target"].shape[-1]
                batch["s1"] = self.streamer.apply_overlap_add_method(batch["s1"], n_chunks)
                batch["s1"] = batch["s1"][:, :length]
                batch["s2"] = self.streamer.apply_overlap_add_method(batch["s2"], n_chunks)
                batch["s2"] = batch["s2"][:, :length]
                batch["s3"] = self.streamer.apply_overlap_add_method(batch["s3"], n_chunks)
                batch["s3"] = batch["s3"][:, :length]
                
                batch["loss"], batch["sisdr"] = self.criterion(**batch, is_train=is_train)

            self.scaler.scale(batch["loss"] / self.grad_accum_step).backward()

            if (batch_idx + 1) % self.grad_accum_step == 0:
                self.scaler.unscale_(self.optimizer)
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        else:
            batch["mix_chunks"], n_chunks = self.streamer.make_chunks(batch["mix"])
            batch = self.move_batch_to_device(batch, self.device, is_train)
            
            batch["s1"], batch["s2"], batch["s3"] = self.model(batch["mix_chunks"], batch["ref"])
            length = batch["target"].shape[-1]
            batch["s1"] = self.streamer.apply_overlap_add_method(batch["s1"], n_chunks)
            batch["s1"] = batch["s1"][:, :length]
            batch["s2"] = self.streamer.apply_overlap_add_method(batch["s2"], n_chunks)
            batch["s2"] = batch["s2"][:, :length]
            batch["s3"] = self.streamer.apply_overlap_add_method(batch["s3"], n_chunks)
            batch["s3"] = batch["s3"][:, :length]
            
            batch["loss"], batch["sisdr"] = self.criterion(**batch, is_train=is_train)
        
        metrics.update("loss", batch["loss"].item())
        metrics.update("SISDR", batch["sisdr"].item())

        for met in metric_names:
            batch[met.lower()] = self.metrics[met](**batch)
            metrics.update(met, batch[met.lower()].item())
        
        return batch
    
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
        return total_norm

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx * self.data_batch_size
        total = self.len_epoch * self.data_batch_size
        return base.format(current, total, 100.0 * current / total)
    
    def _normalize_audio(self, audio):
        factor = torch.max(torch.abs(audio)).item()
        if factor == 0:
            factor = 1
        audio = audio / factor
        return audio
    
    def _log_scalars(self, cur_result):
        for metric_name, metric_value in cur_result.items():
            self.writer.add_scalar(f"{metric_name}", metric_value)

    def _log_sample(self, batch):
        ind = np.random.choice(batch["mix"].shape[0])
        mix_len = int(batch["lens"][ind].item())
        mix = self._normalize_audio(batch["mix"][ind][:mix_len].type(torch.float32))
        ref = self._normalize_audio(batch["ref"][ind].type(torch.float32))
        target = self._normalize_audio(batch["target"][ind][:mix_len].type(torch.float32))
        pred = self._normalize_audio(batch["s1"][ind][:mix_len].type(torch.float32))        
        self.writer.add_audio("mix_audio", mix, self.sr)
        self.writer.add_audio("ref_audio", ref, self.sr)
        self.writer.add_audio("target_audio", target, self.sr)
        self.writer.add_audio("pred_audio", pred, self.sr)

