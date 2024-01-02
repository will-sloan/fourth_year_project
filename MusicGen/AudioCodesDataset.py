# Uses the MyAudioDataset to precompute the compressed audio for all the samples
# Hopefulyl speeding up the training process

from MyAudioDataset import MyAudioDataset
from torch.utils.data import Dataset
import torchaudio
import torch
import pickle


class AudioCodesDataset(Dataset):

    def __init__(self, dataset: MyAudioDataset=None):
        self.audio_dataset = dataset
        self.data_map = []
        
        self.data_dir = dataset.get_data_dir() if dataset is not None else None

    
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self):
        pass

    # Used to precompute the compressed audio for all the samples
    def run_compression(self):
        # Load each sample
        # the __getitem__ function compresses the audio so we just need to save it
        if self.audio_dataset is not None:
            for target, orig, label, sr in self.audio_dataset:
                self.data_map.append({
                    "target": target,
                    "original": orig,
                    "label": label,
                    "sr": sr
                })
                

    def save_data(self, target_file):
        # Save the data map to a file using pickle
        with open(target_file, 'wb') as f:
            pickle.dump(self.data_map, f)

    def load_data(self, target_file):
        # Load the data map from a file using pickle
        with open(target_file, 'rb') as f:
            self.data_map = pickle.load(f)
        