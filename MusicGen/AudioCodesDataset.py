# Uses the MyAudioDataset to precompute the compressed audio for all the samples
# Hopefulyl speeding up the training process

from MyAudioDataset import MyAudioDataset
from torch.utils.data import Dataset
import torchaudio
import torch
import pickle


class AudioCodesDataset(Dataset):

    def __init__(self, comp_model, dataset: MyAudioDataset=None):
        self.audio_dataset = dataset
        self.data_map = []
        self.comp_model = comp_model
        self.data_dir = dataset.get_data_dir() if dataset is not None else None

    
    def __len__(self):
        return len(self.data_map)
    
    def __getitem__(self, index):
        temp = self.data_map[index]
        return temp["target"], temp["target_norm"], temp["original"], temp["original_norm"], temp["label"], temp["sr"]

    # Used to precompute the compressed audio for all the samples
    def run_compression(self):
        # Load each sample
        # the __getitem__ function returns the wav
        # We then need to compress it then save it
        if self.audio_dataset is not None:
            for target, orig, label, sr in self.audio_dataset:
                target = self.compress(target.unsqueeze(0)).squeeze(0)
                orig = self.compress(orig.unsqueeze(0)).squeeze(0)
                tm, ts = torch.mean(target.float()), torch.std(target.float())
                om, os = torch.mean(orig.float()), torch.std(orig.float())


                self.data_map.append({
                    "target": target,
                    "target_norm": (target - tm) / ts,
                    "original": orig,
                    "original_norm": (orig - om) / os,
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

    def compress(self, stereo):
            if self.comp_model is None:
                raise Exception("No compression model found")
            stereo = stereo.cuda()
            with torch.no_grad():
                stereo, scale = self.comp_model.encode(stereo)
            return stereo


    def decompress(self, stereo):
            if self.comp_model is None:
                raise Exception("No compression model found")
            stereo = stereo.cuda()
            with torch.no_grad():
                stereo = self.comp_model.decode(stereo)
            return stereo
        