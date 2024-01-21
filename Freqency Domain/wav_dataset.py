# Choose which side of data to load


# Uses the MyAudioDataset to precompute the compressed audio for all the samples
# Hopefulyl speeding up the training process

from torch.utils.data import Dataset
import torchaudio
import torch
import pickle
import numpy as np
import librosa
import os
import random
from pympler import asizeof

class WavDataset(Dataset):

    def __init__(self, tarpath, basepath, angle, num):
        self.data_map = []

        target_wav, sample_rate = torchaudio.load(tarpath)

        mono_wav, sample_rate = torchaudio.load(basepath)
        print("imported files")

        width = 32000 // 2
        start = width
        end_index = (target_wav.shape[1] - 1) - 16001
        label = float(angle)

        target_wav = target_wav.permute(1, 0)
        mono_wav = mono_wav.permute(1, 0)

        for i in range(num):
            real_index = random.randint(2* width, end_index)
            v = target_wav[real_index]
            window = mono_wav[start-width+real_index:start+width+real_index+1].clone().detach()
            # [0, 32000]
            # print(window.shape)
            print(real_index, v, label, window.shape)
            self.data_map.append(
                {
                    'target_index': real_index,
                    "target_value": v,
                    "angle": label,
                    "input": window,
                }
            )
            with open('/workspace/extension/unet/', 'wb') as f:
                pickle.dump(self.data_map, f)
            print(len(self.data_map))
            size_in_bytes = asizeof.asizeof(self.data_map)

            print(f'Deep size of dictionary: {size_in_bytes} bytes')
            break
            
        print("done random")


    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["input"], temp["angle"], temp["target_value"], temp["target_index"]
    
    def save_data_map(self, path):
        # Add all instance variables to a dictionary
        print(len(self.data_map))
        with open(path, 'wb') as f:
            pickle.dump(self.data_map, f)
    # def my_load(self, tarpath, basepath, angle):
    #     target_wav, sample_rate = torchaudio.load(tarpath)
    #     # right_spec = torch.load(right_ds)
    #     mono_wav, sample_rate = torchaudio.load(basepath)
    #     #print("mono", mono_wav.shape, "target", target_wav.shape)
    #     width = 32000 // 2
    #     start = width
    #     end_index = (target_wav.shape[1] - 1) - 16001
    #     label = float(angle)
    #     #print("start", start, "end", end_index, "label", label)
    #     i = 0
    #     target_wav = target_wav.permute(1, 0)
    #     mono_wav = mono_wav.permute(1, 0)
    #     while i <= end_index:
    #         real_index = i + width
    #         v = target_wav[real_index]
    #         window = mono_wav[start-width+i:start+width+i+1].clone().detach()
    #         # [0, 32000]
    #         self.data_map.append(
    #             {
    #                 'target_index': real_index,
    #                 "target_value": v,
    #                 "angle": label,
    #                 "input": window,
    #             }
    #         )
    #         i += 1
    
    # def random_load(self, tarpath, basepath, angle, num):
    #     self.data_map = []
    #     # Loads num amount of random samples from the wav file
    #     target_wav, sample_rate = torchaudio.load(tarpath)

    #     mono_wav, sample_rate = torchaudio.load(basepath)
    #     print("imported files")

    #     width = 32000 // 2
    #     start = width
    #     end_index = (target_wav.shape[1] - 1) - 16001
    #     label = float(angle)

    #     target_wav = target_wav.permute(1, 0)
    #     mono_wav = mono_wav.permute(1, 0)

    #     for i in range(num):
    #         real_index = random.randint(2* width, end_index)
    #         v = target_wav[real_index]
    #         window = mono_wav[start-width+real_index:start+width+real_index+1].clone().detach()
    #         # [0, 32000]
    #         # print(window.shape)
    #         self.data_map.append(
    #             {
    #                 'target_index': real_index,
    #                 "target_value": v,
    #                 "angle": label,
    #                 "input": window,
    #             }
    #         )
    #     print("done random")

    
    # def load_data_map(self, path):

    #     with open(path, 'rb') as f:
    #         self.data_map = pickle.load(f)

    # def save_random(self, path, num):
    #     # Saves num amount of samples from data_map
    #     # Randomly selected
    #     random.shuffle(self.data_map)
    #     with open(path, 'wb') as f:
    #         pickle.dump(self.data_map[:num], f)
        

        

    