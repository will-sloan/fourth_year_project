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

    def __init__(self, data_dir='/workspace/extension/1sec_spectrogram_files', baseline='recording_01_'):
        self.data_dir = data_dir
        self.data_map = []
        # Load files in data_dir
        for f in os.listdir(data_dir):
            name, ext = os.path.splitext(f)
            # Look at the right spectrograms
            if 'recording' in name or 'EARS_1' in name:
                continue
            else:
                #print(name)
                # Get the proper file names for left,right,orig
                temp = name.split('_')
                index_of_target = name.split('_')[-1]
                #left_ear = temp[0] + '_' + temp[1] + '_' + '1' + '_' + index_of_target + '.npy'
                right_ear = temp[0] + '_' + temp[1] + '_' + '2' + '_' + index_of_target + '.npy'
                 
                orig = baseline + index_of_target + ".npy"
                #print(baseline, index_of_target, os.path.join(data_dir, orig))
                # Load file
                #left_target = np.load(os.path.join(data_dir, left_ear))
                right_target = np.load(os.path.join(data_dir, right_ear))
                orig = np.load(os.path.join(data_dir, orig))
                label = '90'
                # Add to data_map
                self.data_map.append(
                    {
                        "target_spec": right_target,
                        "orig_spec": orig,
                        "label": label,
                    }
                )

    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["left_target"], temp["right_target"], temp["original"], temp["label"]

    