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

    def __init__(self, mono_dir, left_dir, right_dir):
        self.mono_dir = mono_dir
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.data_map = []
    
    def process(self):
        # Load files in data_dir
        print(self.mono_dir)
        print(self.left_dir)
        print(self.right_dir)
        for f in os.listdir(self.left_dir):
            # left_angle_1.npy
            index = f.split('_')[-1].replace('.npy', '')
            angle = f.split('_')[1]
            # Create file names for right and mono
            mono = f'mono_{index}.npy'
            right = f'right_{angle}_{index}.npy'
            #print(baseline, index_of_target, os.path.join(data_dir, orig))
            # Load spectrograms
            left_spec = np.load(os.path.join(self.left_dir, f))
            right_spec = np.load(os.path.join(self.right_dir, right))
            mono_spec = np.load(os.path.join(self.mono_dir, mono))
            label = angle
            # Add to data_map
            self.data_map.append(
                {
                    # "target_spec": right_target,
                    "mono": mono_spec,
                    "label": label,
                    "right": right_spec,
                    "left": left_spec,
                }
            )

    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["mono"], temp["label"], temp["right"], temp["left"]
    
    def save_data_map(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data_map, f)
    
    def load_data_map(self, path):
        with open(path, 'rb') as f:
            self.data_map = pickle.load(f)

    