import os

import warnings

import hydra

import numpy as np

from src.mixer import LibriSpeechSpeakerFiles, MixtureGenerator


warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    for part in ["train", "dev", "test"]:
        directory = os.path.join(config.audios_dir, config[part]["part"])
        speakers_ids = []
        speakers_files = []

        for speaker_id in os.scandir(directory):
            speakers_ids.append(speaker_id.name)
            speakers_files.append(LibriSpeechSpeakerFiles(speaker_id.name, directory, audioTemplate=config.audio_template))

        if part == "train":
            speakers_ids = speakers_ids[: config[part]["max_n_speakers"]]
            speakers_files = speakers_files[: config[part]["max_n_speakers"]]

        mix_generator = MixtureGenerator(speakers_files, 
                                         config[part]["out_folder"], 
                                         nfiles=config[part]["nfiles"], 
                                         test=config[part]["test"], 
                                         randomState=config.random_state)
        mix_generator.generate_mixes(**config.mix_generation_args)


if __name__ == "__main__":
    main()
