import os

import pandas as pd
from tqdm import tqdm

import torch
import torchaudio

import wandb

from ss.wandb import WanDBWriter
from ss.utils import MetricTracker


class Inferencer:
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 dataloader,
                 config,
                 device,
                 logger=None,
                 test_mode=False,
                 criterion=None,
                 metrics=None, 
                 save_inference=False,
                 save_dir=None):
        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.dataloader = dataloader

        self.config = config
        self.device = device

        if rank == 0:
            self.logger = logger
            self.writer = WanDBWriter(
                config, self.logger
            )    
            self.log_step = config["log_step"]
        self.sr = config["dataset"]["sr"]
        if rank == 0 and test_mode:
            self.df = pd.DataFrame(columns=["mix", "ref", "target", "pred"])
            self.df_idx = 0
        elif rank == 0 and not test_mode:
            self.df = pd.DataFrame(columns=["mix", "ref", "pred"])
            self.df_idx = 0
    
        self.test_mode = test_mode
        if test_mode is True:
            self.criterion = criterion
            self.metrics = metrics
            self.metrics_names = [met_name for met_name in metrics.keys() if met_name != "CompositeMetric"]
            if "CompositeMetric" in metrics.keys():
                self.metrics_names += [met_name.upper() for met_name in ["csig", "cbak","covl", "pesq", "ssnr"]]
            self.metric_tracker = MetricTracker("loss", "SI-SDR", *sorted(self.metrics_names), device=device)
            if rank == 0:
                 self.df_values = pd.DataFrame(columns=["loss", "SI-SDR", *sorted(self.metrics_names)])

        self.save_inference = save_inference
        if save_inference:
            self.save_dir = save_dir
            if rank == 0 and not os.path.exists(save_dir):
                os.makedirs(save_dir)

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def move_batch_to_device(batch, device, test_mode):
        for tensor_for_gpu in ["mix", "ref", "lens"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device, non_blocking=True)
        if test_mode:
            batch["target"] = batch["target"].to(device, non_blocking=True)
        return batch
    
    def _normalize_audio(self, audio):
        factor = torch.max(torch.abs(audio)).item()
        if factor == 0:
            factor = 1
        audio = audio / factor
        return audio

    @torch.no_grad()
    def run(self):
        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, desc="inference", total=len(self.dataloader)) if self.rank == 0 else self.dataloader
        ):
            batch = self.move_batch_to_device(batch, self.device, self.test_mode)
            batch["s1"], batch["s2"], batch["s3"] = self.model(batch["mix"], batch["ref"], False)

            if self.test_mode:
                batch["loss"], batch["si-sdr"] = self.criterion(**batch, have_relevant_speakers=False)
                self.metric_tracker.update("loss", batch["loss"].item())
                self.metric_tracker.update("SI-SDR", batch["si-sdr"].item())
                for met in self.metrics.keys():
                    met_value = self.metrics[met](**batch)
                    if isinstance(met_value, dict):
                        for key, value in met_value.items():
                            self.metric_tracker.update(key.upper(), value.item())
                    else:
                        self.metric_tracker.update(met, met_value.item())
            
            if self.rank == 0 and batch_idx % self.log_step == 0 and self.test_mode:
                self.df.loc[self.df_idx] = [
                    wandb.Audio(self._normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["target"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=self.sr) 
                ]
                self.df_idx += 1
            elif self.rank == 0 and batch_idx % self.log_step == 0 and not self.test_mode:
                self.df.loc[self.df_idx] = [
                    wandb.Audio(self._normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=self.sr) 
                ]
                self.df_idx += 1
            
            if self.save_inference:
                torchaudio.save(
                    os.path.join(self.save_dir, batch["pred_name"][0]), 
                    self._normalize_audio(batch["s1"]).detach().cpu(), 
                    sample_rate=self.sr
                )

        if self.rank == 0:
            self.writer.add_table("inference_results", self.df)
            self.logger.info('Inference results are added to wandb')
        
        if self.test_mode:
            results = self.metric_tracker.result_sync()
            if self.rank == 0:
                vals = []
                for metric_name in ["loss", "SI-SDR", *sorted(self.metrics_names)]:
                    metric_value = results[metric_name]
                    self.logger.info("{}: {:.6f}".format(metric_name, metric_value))
                    vals.append(metric_value)
                self.df_values.loc[0] = vals
                self.writer.add_table("inference_values", self.df_values)
                self.logger.info('Inference values are added to wandb')


class CausalInferencer:
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 dataloader,
                 streamer,
                 config,
                 device,
                 logger=None,
                 test_mode=False,
                 criterion=None,
                 metrics=None, 
                 save_inference=False,
                 save_dir=None):
        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.dataloader = dataloader
        self.streamer = streamer

        self.config = config
        self.device = device

        if rank == 0:
            self.logger = logger
            self.writer = WanDBWriter(
                config, self.logger
            )    
            self.log_step = config["log_step"]
        self.sr = config["dataset"]["sr"]
        if rank == 0 and test_mode:
            self.df = pd.DataFrame(columns=["mix", "ref", "target", "pred"])
            self.df_idx = 0
        elif rank == 0 and not test_mode:
            self.df = pd.DataFrame(columns=["mix", "ref", "pred"])
            self.df_idx = 0
    
        self.test_mode = test_mode
        if test_mode is True:
            self.criterion = criterion
            self.metrics = metrics
            self.metrics_names = [met_name for met_name in metrics.keys() if met_name != "CompositeMetric"]
            if "CompositeMetric" in metrics.keys():
                self.metrics_names += [met_name.upper() for met_name in ["csig", "cbak","covl", "pesq", "ssnr"]]
            self.metric_tracker = MetricTracker("loss", "SI-SDR", *sorted(self.metrics_names), device=device)
            if rank == 0:
                 self.df_values = pd.DataFrame(columns=["loss", "SI-SDR", *sorted(self.metrics_names)])

        self.save_inference = save_inference
        if save_inference:
            self.save_dir = save_dir
            if rank == 0 and not os.path.exists(save_dir):
                os.makedirs(save_dir)

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def move_batch_to_device(batch, device, test_mode):
        for tensor_for_gpu in ["mix_chunks", "ref", "lens"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device, non_blocking=True)
        if test_mode:
            batch["target"] = batch["target"].to(device, non_blocking=True)
        return batch
    
    def _normalize_audio(self, audio):
        factor = torch.max(torch.abs(audio)).item()
        if factor == 0:
            factor = 1
        audio = audio / factor
        return audio

    @torch.no_grad()
    def run(self):
        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, desc="inference", total=len(self.dataloader)) if self.rank == 0 else self.dataloader
        ):
            batch["mix_chunks"], n_chunks = self.streamer.make_chunks(batch["mix"])
            batch = self.move_batch_to_device(batch, self.device, self.test_mode)
            batch["s1"] = []
            batch["s2"] = []
            batch["s3"] = []
            for i in range(n_chunks):
                chunk = batch["mix"][i: i + 1, :]
                if i == 0:
                    s1, s2, s3, ref_vec = self.model(chunk, batch["ref"], False, one_chunk=True)
                    batch["s1"].append(s1)
                    batch["s2"].append(s2)
                    batch["s3"].append(s3)
                else:
                    s1, s2, s3 = self.model(chunk, ref=None, have_relevant_speakers=False, 
                                            ref_vec=ref_vec, one_chunk=True)
                    batch["s1"].append(s1)
                    batch["s2"].append(s2)
                    batch["s3"].append(s3)
            batch["s1"] = torch.cat(batch["s1"], dim=0)
            batch["s2"] = torch.cat(batch["s2"], dim=0)
            batch["s3"] = torch.cat(batch["s3"], dim=0)
            length = batch["lens"][0]
            batch["s1"] = self.streamer.apply_overlap_add_method(batch["s1"], n_chunks)
            batch["s1"] = batch["s1"][:, :length]
            batch["s2"] = self.streamer.apply_overlap_add_method(batch["s2"], n_chunks)
            batch["s2"] = batch["s2"][:, :length]
            batch["s3"] = self.streamer.apply_overlap_add_method(batch["s3"], n_chunks)
            batch["s3"] = batch["s3"][:, :length]

            if self.test_mode:
                batch["loss"], batch["si-sdr"] = self.criterion(**batch, have_relevant_speakers=False)
                self.metric_tracker.update("loss", batch["loss"].item())
                self.metric_tracker.update("SI-SDR", batch["si-sdr"].item())
                for met in self.metrics.keys():
                    met_value = self.metrics[met](**batch)
                    if isinstance(met_value, dict):
                        for key, value in met_value.items():
                            self.metric_tracker.update(key.upper(), value.item())
                    else:
                        self.metric_tracker.update(met, met_value.item())
            
            if self.rank == 0 and batch_idx % self.log_step == 0 and self.test_mode:
                self.df.loc[self.df_idx] = [
                    wandb.Audio(self._normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["target"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=self.sr) 
                ]
                self.df_idx += 1
            elif self.rank == 0 and batch_idx % self.log_step == 0 and not self.test_mode:
                self.df.loc[self.df_idx] = [
                    wandb.Audio(self._normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=self.sr),
                    wandb.Audio(self._normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=self.sr) 
                ]
                self.df_idx += 1
            
            if self.save_inference:
                torchaudio.save(
                    os.path.join(self.save_dir, batch["pred_name"][0]), 
                    self._normalize_audio(batch["s1"]).detach().cpu(), 
                    sample_rate=self.sr
                )

        if self.rank == 0:
            self.writer.add_table("inference_results", self.df)
            self.logger.info('Inference results are added to wandb')
        
        if self.test_mode:
            results = self.metric_tracker.result_sync()
            if self.rank == 0:
                vals = []
                for metric_name in ["loss", "SI-SDR", *sorted(self.metrics_names)]:
                    metric_value = results[metric_name]
                    self.logger.info("{}: {:.6f}".format(metric_name, metric_value))
                    vals.append(metric_value)
                self.df_values.loc[0] = vals
                self.writer.add_table("inference_values", self.df_values)
                self.logger.info('Inference values are added to wandb')
