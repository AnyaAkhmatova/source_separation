import os

import numpy as np

import librosa
import soundfile as sf

import warnings


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)
    noise_norm = noise * (amp_noise / np.linalg.norm(noise))
    mix = clean + noise_norm

    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mix.max(axis=0) > max_int16 or mix.min(axis=0) < min_int16:
        if mix.max(axis=0) >= abs(mix.min(axis=0)): 
            reduction_rate = max_int16 / mix.max(axis=0)
        else:
            reduction_rate = min_int16 / mix.min(axis=0)
        mix = mix * reduction_rate

    return mix

def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)

def cut_audios(s1, s2, sec, sr):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)
    s1_cut = []
    s2_cut = []
    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])
        segment += 1
    return s1_cut, s2_cut

def fix_length(s1, s2, min_or_max='max'):
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2

def create_mix(idx, triplet, snr_levels, out_dir, test=False, sr=16000, **kwargs):
    trim_db, vad_db = kwargs["trim_db"], kwargs["vad_db"]
    use_vad_merge = kwargs.get("use_vad_merge", False)

    audioLen = kwargs["audio_len"]
    refMaxLen = kwargs["ref_max_len"] * sr

    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    amp_s1 = np.max(np.abs(s1))
    amp_s2 = np.max(np.abs(s2))
    amp_ref = np.max(np.abs(ref))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(ref, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2, top_db=trim_db)

    if len(ref) < sr:
        return

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")

    if not test:
        ref = ref[:refMaxLen]

        if use_vad_merge:
            s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audioLen, sr)

        for i in range(len(s1_cut)):
            snr = np.random.choice(snr_levels, 1).item()
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            norm = np.max(np.abs(mix)) * 1.1
            s1_cut[i] /= norm
            mix /= norm

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)

    else:
        s1, s2 = fix_length(s1, s2, 'min')
        snr = np.random.choice(snr_levels, 1).item()
        mix = snr_mixer(s1, s2, snr)

        norm = np.max(np.abs(mix)) * 1.1

        if norm == 0:
            warnings.warn(f"Norm is zero for target:{s1_path} and noise: {s2_path}. Len of s1 - {len(s1)}, len of s2 - {len(s2)}")
            return

        s1 /= norm
        mix /= norm

        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)
