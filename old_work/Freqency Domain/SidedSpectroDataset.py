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

class SidedSpectroDataset(Dataset):

    def __init__(self, mono_dir, spec_dir, chunk_size=1000):
        self.mono_dir = mono_dir
        self.spec_dir = spec_dir
        self.data_map = []

        self.chunk_size = chunk_size
        self.file_list = os.listdir(self.spec_dir)
        random.shuffle(self.file_list) 
        self.current_index = 0

    def load_next_chunk(self):
        chunk_data = []

        for i in range(self.current_index, min(self.current_index + self.chunk_size, len(self.file_list))):
            # left_angle_1.npy
            f = self.file_list[i]
            index = f.split('_')[-1].replace('.pt', '')

            # There are 3419 mono files, so check to make sure the index is valid
            if int(index) > 3419:
                continue

            angle = f.split('_')[1]
            # Create file names for right and mono
            mono = f'mono_{index}'

            # Load spectrograms
            ds = os.path.join(self.spec_dir, f)
            mono_ds = os.path.join(self.mono_dir, mono)
            # print(ds)
            # print(mono_ds)
            # spec = torch.from_numpy(np.load(ds))
            spec = torch.load(ds)

            # mono_spec = torch.from_numpy(np.load(mono_ds))
            mono_spec = torch.load(mono_ds)
            label = float(angle)
            # Add to data_map
            chunk_data.append(
                {
                    "mono": mono_spec,
                    "label": label,
                    "target": spec,
                }
            )
        self.current_index += self.chunk_size
        return chunk_data
    
    # Get one chunk instead of all the chunks
    def load_chunk(self):
        #print(self.current_index, len(self.file_list))
        if self.current_index >= len(self.file_list):
            print("No more new chunks to load, resetting index")
            self.current_index = 0
        self.data_map = self.load_next_chunk()
        print("Done loading chunk")
    
    def process_all(self):
        # Load files in data_dir
        while self.current_index < len(self.file_list):
            chunk_data = self.load_next_chunk()
            self.data_map.extend(chunk_data)
        print("Done loading all data")
    
    def normalize(self):
        # Only normalizes the current stores chunk

        self.mean_value /= self.total_samples
        for data in self.data_map:
            self.std_value += ((np.abs(data['target']) - self.mean_value) ** 2).sum()
            self.std_value += ((np.abs(data['mono']) - self.mean_value) ** 2).sum()
        self.std_value = np.sqrt(self.std_value / self.total_samples)

        # Normalize data
        for data in self.data_map:
            data['target'] = (np.abs(data['target']) - self.mean_value) / self.std_value
            data['mono'] = (np.abs(data['mono']) - self.mean_value) / self.std_value
        print("Done normalizing data")

    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["mono"], temp["label"], temp["target"]
    
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
            "spec_dir": self.spec_dir,
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
        self.spec_dir = temp["spec_dir"]

    