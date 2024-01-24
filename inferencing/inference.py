import os
import glob
from glob import glob
from pathlib import Path
import logging 
import warnings

import hydra
import wandb

import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

import ss.model as module_arch
from ss.utils import init_obj

warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    wandb.login()
    if config.get('wandb_project') is None:
        raise ValueError("please specify project name for wandb")
    wandb.init(
        project=config['wandb_project'],
        config=dict(config)
    )
    
    save_dir = config["output_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    load_dir = config["input_dir"]
    mixes = sorted(glob(os.path.join(load_dir, '*-mixed.wav')))
    refs = sorted(glob(os.path.join(load_dir, '*-ref.wav')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(device)

    model = init_obj(config["arch"], module_arch, n_speakers=config["n_speakers"]).to(device)
    logger.info(model)

    if config.get('resume') is None:
        raise ValueError("please specify resume path")
    logger.info("Loading checkpoint: {} ...".format(config['resume']))
    checkpoint = torch.load(config['resume'], device)
    if checkpoint["config"]["arch"] != config["arch"]:
        logger.warning(
            "Warning: Architecture configuration given in config file is different from that "
            "of checkpoint. This may yield an exception while state_dict is being loaded."
        )
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(
        "Checkpoint loaded."
    )

    log_step = config["log_step"]
    sr = config["sr"]
    df = pd.DataFrame(columns=["mix", "ref", "pred"])
    df_idx = 0

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(mixes)), desc="inferencing", total=len(mixes)):
            mix, _ = torchaudio.load(mixes[idx])
            ref, _ = torchaudio.load(refs[idx])
            batch = {"mix": mix, "ref": ref}
            for tensor_for_gpu in ["mix", "ref"]:
                batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
            s1, _, _, _ = model(batch["mix"], batch["ref"])
            if idx % log_step == 0:
                df.loc[df_idx] = [
                    wandb.Audio(batch["mix"][0].detach().cpu().numpy().T, sample_rate=sr),
                    wandb.Audio(batch["ref"][0].detach().cpu().numpy().T, sample_rate=sr),
                    wandb.Audio(s1[0].detach().cpu().numpy().T, sample_rate=sr) 
                ]
                df_idx += 1
            torchaudio.save(
                os.path.join(save_dir, mixes[idx].split("/")[-1].split("-")[0] + '-pred.wav'), 
                s1.detach().cpu(), 
                sample_rate=sr
            )
    
    wandb.log({"inference_results": wandb.Table(dataframe=df)})

    logger.info('Inference results are added to wandb')

if __name__ == "__main__":
    main()

