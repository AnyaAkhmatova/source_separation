import logging 
import warnings

import numpy as np
import random
from tqdm import tqdm
import time

import torch

import onnx
import onnxruntime
from onnx_opcounter import calculate_params

import hydra

from ss.datasets import get_simple_dataloader
from ss.streamer import Streamer

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)
random.seed(SEED)


def run(config, logger, device):    
    dataloader = get_simple_dataloader(**config["dataset"]["test"])
    assert config["dataset"]["test"]["batch_size"] == 1, "batch_size must be 1"

    streamer = Streamer(**config["streamer"])

    gpu = (config["mode"] == "gpu")

    speaker_handler_session = onnxruntime.InferenceSession(config["speaker_handler_path"], 
                                                           providers=['CUDAExecutionProvider' if gpu else 'CPUExecutionProvider'])
    main_model_session = onnxruntime.InferenceSession(config["main_model_path"], 
                                                      providers=['CUDAExecutionProvider' if gpu else 'CPUExecutionProvider'])

    model = onnx.load_model(config["speaker_handler_path"])
    params = calculate_params(model)
    logger.info("Number of speaker handler parameters:" + str(params))
    model = onnx.load_model(config["main_model_path"])
    params = calculate_params(model)
    logger.info("Number of main model parameters:" + str(params))

    num_iters = config["num_iters"]
    times = []
    mix_lengths = []
    ref_lengths = []
    chunks_nums = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="measuring_gpu_time", total=num_iters)):
        if batch_idx + 1 >= num_iters:
            break

        mix, ref = batch["mix"], batch["ref"]
        mix_chunks, n_chunks = streamer.make_chunks(mix)
        mix_chunks = mix_chunks.to(device)
        ref = ref.to(device)

        if gpu:
            start = time.time()

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

            end = time.time()
        
        else:

            start = time.time()

            ref_vec, logits = speaker_handler_session.run(None, {"ref": ref.numpy()})
            ref_vec = ref_vec.numpy()

            batch["s1"] = []
            memory = torch.zeros((1, config["main_model"]["memory_size"], config["time_dim"]), 
                                 dtype=torch.float32, device=device)
            for i in range(n_chunks):
                chunk = mix_chunks[i: i + 1, :]

                s1_chunk, new_memory = main_model_session.run(None, {"chunk": chunk.numpy(), "ref_vec": ref_vec, "memory": memory.numpy()})

                batch["s1"].append(torch.tensor(s1_chunk))
                memory = new_memory
            
            batch["s1"] = torch.cat(batch["s1"], dim=0)

            end = time.time()

        times.append(end - start)
        mix_lengths.append(mix.shape[-1]/config["dataset"]["sr"])
        ref_lengths.append(ref.shape[-1]/config["dataset"]["sr"])
        chunks_nums.append(n_chunks)

    mean_time = np.array(times).mean()
    mean_mix_length = np.array(mix_lengths).mean()
    mean_ref_length = np.array(ref_lengths).mean()
    mean_chunks_num = np.array(chunks_nums).mean()
    if gpu:
        logger.info(f"Average gpu time per sample (s): {mean_time}")
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

