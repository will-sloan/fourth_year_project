# Uses the MyAudioDataset to precompute the compressed audio for all the samples
# Hopefulyl speeding up the training process

from torch.utils.data import Dataset
import torchaudio
import torch
import pickle
import numpy as np
import librosa
import os


class SpectroDataset(Dataset):

    def __init__(self, data_dir='/workspace/extension/unet', baseline='mono', left = 'left', right = 'right'):
        self.data_dir = data_dir
        self.data_map = []
        # Load files in data_dir
        for f in os.listdir(os.path.join(data_dir, baseline)):
            # mono_angle_1.npy
            index = f.split('_')[-1]
            # Create file names for left and right
            left_file = left + '_' + index + '.npy'
            right_file = right + '_' + index + '.npy'
            #print(baseline, index_of_target, os.path.join(data_dir, orig))
            # Load file
            left_target = np.load(os.path.join(data_dir, left, left_file))
            right_target = np.load(os.path.join(data_dir, right, right_file))
            orig = np.load(os.path.join(data_dir, f))
            label = f.split('_')[1]
            # Add to data_map
            self.data_map.append(
                {
                    # "target_spec": right_target,
                    "orig": orig,
                    "label": label,
                    "right_ear": right_target,
                    "left_ear": left_target,
                }
            )

    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["orig"], temp["label"], temp["right_ear"], temp["left_ear"]

    