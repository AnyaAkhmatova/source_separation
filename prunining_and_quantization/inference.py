import os
import logging 
import warnings

import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torchaudio

import onnxruntime

import hydra

import wandb

import ss.loss as module_loss
import ss.metric as module_metric
from ss.utils import init_obj, MetricTracker
from ss.datasets import get_simple_dataloader
from ss.streamer import Streamer

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def normalize_audio(audio):
    factor = torch.max(torch.abs(audio)).item()
    if factor == 0:
        factor = 1
    audio = audio / factor
    return audio


def run_inferencing(config, logger):
    wandb.login()
    if config.get('wandb_project') is None:
        raise ValueError("Please specify project name for wandb")
    wandb.init(
        project=config['wandb_project'],
        name=config['name'],
        config=dict(config)
    )

    device = torch.device("cuda:0")
    
    dataloader = get_simple_dataloader(**config["dataset"]["inference"])

    streamer = Streamer(**config["streamer"])

    speaker_handler_session = onnxruntime.InferenceSession(config["speaker_handler_path"], 
                                                           providers=['CUDAExecutionProvider'])
    main_model_session = onnxruntime.InferenceSession(config["main_model_path"], 
                                                      providers=['CUDAExecutionProvider'])

    test_mode = config["data"]["inference"]["test_mode"]
    save_inference = config["save_inference"]
    if save_inference:
        save_dir = config["save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if test_mode:
        criterion = init_obj(config["loss"], module_loss).to(device)
        metrics = [
            init_obj(metric_dict, module_metric, device=device)
            for metric_dict in config["metrics"]
        ]
        metrics = {met.name: met for met in metrics}
        metrics_names = [met_name for met_name in metrics.keys() if met_name != "CompositeMetric"]
        if "CompositeMetric" in metrics.keys():
            metrics_names += [met_name.upper() for met_name in ["csig", "cbak","covl", "pesq", "ssnr"]]
        metric_tracker = MetricTracker("loss", "SI-SDR", *sorted(metrics_names), device=device)
        df = pd.DataFrame(columns=["mix", "ref", "target", "pred"])
        df_idx = 0
        df_vals = pd.DataFrame(columns=["loss", "SI-SDR", *sorted(metrics_names)])
    else:
        df = pd.DataFrame(columns=["mix", "ref", "pred"])
        df_idx = 0     
        
    log_step = config["log_step"]
    sr = config["sr"]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="testing", total=len(dataloader))):
        batch["mix_chunks"], n_chunks = streamer.make_chunks(batch["mix"])
        for tensor_for_gpu in (["mix_chunks", "ref", "target", "lens"] if test_mode else ["mix_chunks", "ref", "lens"]):
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)

        binding = speaker_handler_session.io_binding()
        ref = batch["ref"].contiguous()
        binding.bind_input(
            name='ref',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(ref.shape),
            buffer_ptr=ref.data_ptr(),
        )
        ref_vec_shape = (1, config["speaker_handler"]["out_channels"])
        ref_vec = torch.empty(ref_vec_shape, dtype=torch.float32, device=device).contiguous()
        binding.bind_output(
            name='ref_vec',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(ref_vec.shape),
            buffer_ptr=ref_vec.data_ptr(),
        )
        logits_shape = (1, config["speaker_handler"]["n_speakers"])
        logits = torch.empty(logits_shape, dtype=torch.float32, device=device).contiguous()
        binding.bind_output(
            name='speaker_logits',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(logits.shape),
            buffer_ptr=logits.data_ptr(),
        )
        speaker_handler_session.run_with_iobinding(binding)

        batch["s1"] = []
        memory = torch.zeros((1, config["main_model"]["memory_size"], config["time_dim"]), 
                             dtype=torch.float32, device=device)
        for i in range(n_chunks):
            chunk = batch["mix_chunks"][i: i + 1, :]

            binding = main_model_session.io_binding()
            chunk = chunk.contiguous()
            binding.bind_input(
                name='chunk',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(chunk.shape),
                buffer_ptr=chunk.data_ptr(),
            )
            ref_vec = ref_vec.contiguous()
            binding.bind_input(
                name='ref_vec',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(ref_vec.shape),
                buffer_ptr=ref_vec.data_ptr(),
            )
            memory = memory.contiguous()
            binding.bind_input(
                name='memory',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(memory.shape),
                buffer_ptr=memory.data_ptr(),
            )
            s1_chunk_shape = (1, config["streamer"]["chunk_window"])
            s1_chunk = torch.empty(s1_chunk_shape, dtype=torch.float32, device=device).contiguous()
            binding.bind_output(
                name='s1_chunk',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(s1_chunk.shape),
                buffer_ptr=s1_chunk.data_ptr(),
            )
            new_memory_shape = (1, config["main_model"]["memory_size"], config["time_dim"])
            new_memory = torch.empty(new_memory_shape, dtype=torch.float32, device=device).contiguous()
            binding.bind_output(
                name='new_memory',
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(new_memory.shape),
                buffer_ptr=new_memory.data_ptr(),
            )
            main_model_session.run_with_iobinding(binding)

            batch["s1"].append(s1_chunk)
            memory = new_memory

        batch["s1"] = torch.cat(batch["s1"], dim=0)
        length = batch["lens"][0]
        batch["s1"] = streamer.apply_overlap_add_method(batch["s1"], n_chunks)
        batch["s1"] = batch["s1"][:, :length]

        if test_mode:
            batch["loss"], batch["si-sdr"] = criterion(**batch, have_relevant_speakers=False)
            metric_tracker.update("loss", batch["loss"].item())
            metric_tracker.update("SI-SDR", batch["si-sdr"].item())
            for met in metrics.keys():
                met_value = metrics[met](**batch)
                if isinstance(met_value, dict):
                    for key, value in met_value.items():
                        metric_tracker.update(key.upper(), value.item())
                else:
                    metric_tracker.update(met, met_value.item())

        if batch_idx % log_step == 0 and test_mode:
            df.loc[df_idx] = [
                wandb.Audio(normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=sr),
                wandb.Audio(normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=sr),
                wandb.Audio(normalize_audio(batch["target"][0]).detach().cpu().numpy().T, sample_rate=sr),
                wandb.Audio(normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=sr) 
            ]
            df_idx += 1
        elif batch_idx % log_step == 0 and not test_mode:
            df.loc[df_idx] = [
                wandb.Audio(normalize_audio(batch["mix"][0]).detach().cpu().numpy().T, sample_rate=sr),
                wandb.Audio(normalize_audio(batch["ref"][0]).detach().cpu().numpy().T, sample_rate=sr),
                wandb.Audio(normalize_audio(batch["s1"][0]).detach().cpu().numpy().T, sample_rate=sr) 
            ]
            df_idx += 1

        if save_inference:
            torchaudio.save(
                os.path.join(save_dir, batch["pred_name"][0]), 
                normalize_audio(batch["s1"]).detach().cpu(), 
                sample_rate=sr
            )

    wandb.log({"inference_results": wandb.Table(dataframe=df)})
    logger.info('Inference results are added to wandb')

    if test_mode:
        results = metric_tracker.result()
        vals = []
        for metric_name in ["loss", "SI-SDR", *sorted(metrics_names)]:
            metric_value = results[metric_name]
            logger.info("{}: {:.6f}".format(metric_name, metric_value))
            vals.append(metric_value)
        df_vals.loc[0] = vals
        wandb.log({"inference_values": wandb.Table(dataframe=df_vals)})
        logger.info('Inference values are added to wandb')


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 1, "Require >= 1 GPU"
    run_inferencing(config, logger)
    

if __name__ == "__main__":
    main()

