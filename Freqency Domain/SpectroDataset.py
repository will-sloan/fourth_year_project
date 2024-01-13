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

    def __init__(self, mono_dir, left_dir, right_dir, chunk_size=1000, max_chunks=100000):
        self.mono_dir = mono_dir
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.data_map = []
        self.min_value = np.inf
        self.max_value = -np.inf
        self.mean_value = 0
        self.std_value = 0
        self.total_samples = 0

        self.chunk_size = chunk_size
        self.file_list = os.listdir(self.left_dir)
        self.current_index = 0
        self.num_chunks = 0
        self.max_chunks = max_chunks

    def load_next_chunk(self):
        chunk_data = []
        print("Loading chunk")
        for i in range(self.current_index, min(self.current_index + self.chunk_size, len(self.file_list))):
            # left_angle_1.npy
            f = self.file_list[i]
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
            left_spec = left_spec[:-1, :-1]
            right_spec = right_spec[:-1, :-1]
            mono_spec = mono_spec[:-1, :-1]
            # Compute magnitude of complex numbers
            left_mag = np.abs(left_spec)
            right_mag = np.abs(right_spec)
            mono_mag = np.abs(mono_spec)
            # Update min, max, mean, std
            self.min_value = min(self.min_value, left_mag.min(), right_mag.min(), mono_mag.min())
            self.max_value = max(self.max_value, left_mag.max(), right_mag.max(), mono_mag.max())
            self.mean_value += left_mag.sum() + right_mag.sum() + mono_mag.sum()
            self.total_samples += left_mag.size + right_mag.size + mono_mag.size

            left_spec = torch.view_as_real(torch.from_numpy(left_spec))
            right_spec = torch.view_as_real(torch.from_numpy(right_spec))
            mono_spec = torch.view_as_real(torch.from_numpy(mono_spec))
            left_spec = left_spec.permute(2, 0, 1)
            right_spec = right_spec.permute(2, 0, 1)
            mono_spec = mono_spec.permute(2, 0, 1)
            label = float(angle)
            # Add to data_map
            chunk_data.append(
                {
                    # "target_spec": right_target,
                    "orig": mono_spec,
                    "mono": mono_spec,
                    "label": label,
                    "right": right_spec,
                    "left": left_spec,

                }
            )
        self.current_index += self.chunk_size
        self.num_chunks += 1
        return chunk_data
    
    def process(self):
        # Load files in data_dir
        while self.current_index < len(self.file_list) and self.num_chunks < self.max_chunks:
            chunk_data = self.load_next_chunk()
            self.data_map.extend(chunk_data)
        print("Done loading data")
    
    def normalize(self):
        self.mean_value /= self.total_samples
        for data in self.data_map:
            self.std_value += ((np.abs(data['left']) - self.mean_value) ** 2).sum()
            self.std_value += ((np.abs(data['right']) - self.mean_value) ** 2).sum()
            self.std_value += ((np.abs(data['mono']) - self.mean_value) ** 2).sum()
        self.std_value = np.sqrt(self.std_value / self.total_samples)

        # Normalize data
        for data in self.data_map:
            data['left'] = (np.abs(data['left']) - self.mean_value) / self.std_value
            data['right'] = (np.abs(data['right']) - self.mean_value) / self.std_value
            data['mono'] = (np.abs(data['mono']) - self.mean_value) / self.std_value
        print("Done normalizing data")

    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["mono"], temp["label"], temp["right"], temp["left"],temp["orig"]
    
    def save_data_map(self, path):
        # Add all instance variables to a dictionary
        temp = {
            "data_map": self.data_map,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "total_samples": self.total_samples,
            "chunk_size": self.chunk_size,
            "file_list": self.file_list,
            "current_index": self.current_index,
            "num_chunks": self.num_chunks,
            "max_chunks": self.max_chunks,
            "mono_dir": self.mono_dir,
            "left_dir": self.left_dir,
            "right_dir": self.right_dir
        }
        with open(path, 'wb') as f:
            pickle.dump(temp, f)
    
    def load_data_map(self, path):

        with open(path, 'rb') as f:
            temp = pickle.load(f)
        
        # Unpack the dictionary
        self.data_map = temp["data_map"]
        self.min_value = temp["min_value"]
        self.max_value = temp["max_value"]
        self.mean_value = temp["mean_value"]
        self.std_value = temp["std_value"]
        self.total_samples = temp["total_samples"]
        self.chunk_size = temp["chunk_size"]
        self.file_list = temp["file_list"]
        self.current_index = temp["current_index"]
        self.num_chunks = temp["num_chunks"]
        self.max_chunks = temp["max_chunks"]
        self.mono_dir = temp["mono_dir"]
        self.left_dir = temp["left_dir"]
        self.right_dir = temp["right_dir"]

    