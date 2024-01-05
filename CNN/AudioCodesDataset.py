# Uses the MyAudioDataset to precompute the compressed audio for all the samples
# Hopefulyl speeding up the training process

from MyAudioDataset import MyAudioDataset
from torch.utils.data import Dataset
import torchaudio
import torch
import pickle
import numpy as np
import librosa


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


    # Augments the data set by adding noise and speed change
    # Does not time shift or pitch shift since that may change the spatial qualities?
    # Does not save the intermediate wav files of the augmented data
    def augment_dataset(self):
        # For each sample in the dataset
        # Add noise
        # Add speed change

        # For each sample in the dataset
        # Load the wav, augment, then save a copy or both it and the target with the same label
        for target, orig, label, sr in self.audio_dataset:
            # Augment the wav
            # Add speed change
            speed_change = np.random.uniform(low=0.8, high=1.2)
            # Convert to numpy on cpu
            orig = orig.cpu().numpy()
            target = target.cpu().numpy()
            orig_speed_change = librosa.effects.time_stretch(y=orig, rate=speed_change)
            target_speed_change = librosa.effects.time_stretch(y=target, rate=speed_change)
            
            print(orig_speed_change.shape, target_speed_change.shape)
            print(orig_speed_change.shape[1])
            # Pad orig and target if they are not 320000
            if orig_speed_change.shape[1] < 320000:
                padding = ((0, 0), (0, 320000 - orig_speed_change.shape[1]))
                orig_speed_change = np.pad(orig_speed_change, padding, mode='constant')

            if target_speed_change.shape[1] < 320000:
                padding = ((0, 0), (0, 320000 - target_speed_change.shape[1]))
                target_speed_change = np.pad(target_speed_change, padding, mode='constant')

            print(orig_speed_change.shape, target_speed_change.shape)
            #cpu_copy = orig_speed_change.copy()
            # Convert back to tensors
            orig_speed_change = torch.from_numpy(orig_speed_change).cuda()
            target_speed_change = torch.from_numpy(target_speed_change).cuda()
            # Convert to codes
            print(orig_speed_change.shape, target_speed_change.shape)
            orig_speed_change = self.compress(orig_speed_change.unsqueeze(0)).squeeze(0)
            target_speed_change = self.compress(target_speed_change.unsqueeze(0)).squeeze(0)

            # Make sure data is 500 long in second dimension
            orig_speed_change = orig_speed_change[:,:500]
            target_speed_change = target_speed_change[:,:500]
            print(orig_speed_change.shape, target_speed_change.shape)

            # Normalise
            tm, ts = torch.mean(target_speed_change.float()), torch.std(target_speed_change.float())
            om, os = torch.mean(orig_speed_change.float()), torch.std(orig_speed_change.float())
            #assert len(orig_speed_change) == 500, f"Length of noise is not 500, {len(orig_speed_change)}"
            self.data_map.append({
                    "target": target_speed_change,
                    "target_norm": (target_speed_change - tm) / ts,
                    "original": orig_speed_change,
                    "original_norm": (orig_speed_change - om) / os,
                    "label": label,
                    "sr": sr
                })
            # Add noise

            #orig = orig.cpu().numpy()
            #target = target.cpu().numpy()
            noise = np.random.randn(*orig.shape)
            print(noise.shape, orig.shape)

            orig_noise = orig + 0.005 * noise
            target_noise = target + 0.005 * noise
            print(orig_noise.shape, target_noise.shape)
            if orig_noise.shape[1] < 320000:
                padding = ((0, 0), (0, 320000 - orig_noise.shape[1]))
                orig_noise = np.pad(orig_noise, padding, mode='constant')

            if target_noise.shape[1] < 320000:
                padding = ((0, 0), (0, 320000 - target_noise.shape[1]))
                target_noise = np.pad(target_noise, padding, mode='constant')
            
            print(orig_noise.shape, target_noise.shape)

            # Convert back to tensors
            orig_noise = torch.from_numpy(orig_noise).cuda().float()
            target_noise = torch.from_numpy(target_noise).cuda().float()
            # Convert to codes
            print(orig_noise.shape, target_noise.shape)
            orig_noise = self.compress(orig_noise.unsqueeze(0)).squeeze(0)
            target_noise = self.compress(target_noise.unsqueeze(0)).squeeze(0)

            # Make sure data is 500 long
            orig_noise = orig_noise[:,:500]
            target_noise = target_noise[:,:500]
            print(orig_noise.shape, target_noise.shape)

            # Normalise
            tm, ts = torch.mean(target_noise.float()), torch.std(target_noise.float())
            om, os = torch.mean(orig_noise.float()), torch.std(orig_noise.float())
            #assert len(orig_noise) == 500, f"Length of noise is not 500, {len(orig_noise)}"
            self.data_map.append({
                    "target": target_noise,
                    "target_norm": (target_noise - tm) / ts,
                    "original": orig_noise,
                    "original_norm": (orig_noise - om) / os,
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
    
    def set_audio_dataset(self, dataset):
        self.audio_dataset = dataset
        self.data_dir = dataset.get_data_dir()
        