import os
import glob
from glob import glob




path = os.path.join('../../data/ss_data_360', 'train')
max_length = 20000

file_names = sorted(glob(os.path.join(path, '*-mixed.wav')))[: max_length]

target_id_data_id = sorted(list(set([file_name.split('/')[-1].split('_')[0] for file_name in file_names])))
data_id_target_id = {target_id_data_id[target_id]: target_id for target_id in range(len(target_id_data_id))}
n_speakers = len(target_id_data_id)

# assert n_speakers == 100

output_file = "../../data/ss_data_360/train_speakers.txt"

with open(output_file, "w") as f:
    for target_id in range(len(target_id_data_id)):
        data_id = target_id_data_id[target_id]
        f.write(f"{target_id} {data_id}\n")

