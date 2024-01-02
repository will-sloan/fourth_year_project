from torch.utils.data import Dataset
import os
import torchaudio
import torch

# Code from: https://github.com/chavinlo/musicgen_trainer/blob/main/train.py
# Create a class to hold your data set
# Modified since we are not loading the same way as the original code. 
class MyAudioDataset(Dataset):
    def __init__(self, data_dir='/workspace/small_model_data2', baseline='recording_01_'):
        # Uses the absolute path to the directory where the data is stored
        self.data_dir = data_dir
        self.data_map = []
        #self.baseline_file_name = baseline_file_name
        dir_map = os.listdir(data_dir)
        for d in dir_map:
            name, ext = os.path.splitext(d)
            # Only have 1 data point for each recording
            # that takes the left/right and the baseline
            if 'recording' in name or 'EARS_1' in name:
                continue
            if ext == ".wav":
                # We will have labels for everything
                #if os.path.exists(os.path.join(data_dir, name + ".txt")):
                label = name.split('Deg')[0]
                if label is not None:
                    temp = name.split('_')
                    index_of_target = name.split('_')[-1]
                    left_ear = temp[0] + '_' + temp[1] + '_' + '1' + '_' + index_of_target + '.wav'
                    right_ear = temp[0] + '_' + temp[1] + '_' + '2' + '_' + index_of_target + '.wav'
                    orig = baseline + index_of_target + ".wav"
                    self.data_map.append(
                        {
                            "left_target": os.path.join(data_dir, left_ear),
                            "right_target": os.path.join(data_dir, right_ear),
                            "label": label,
                            "original": os.path.join(data_dir, orig),
                        }
                    )
                else:
                    raise ValueError(f"No label file for {name}")

    def __len__(self):
        return len(self.data_map)
    
    def get_data_dir(self):
        return self.data_dir

    # def __getitem__(self, idx):
    #     data = self.data_map[idx]
    #     left_audio = data["left_target"]
    #     right_audio = data["right_target"]
    #     label = data.get("label", "")
    #     original = data.get("original", "")

    #     return left_audio, right_audio, label, original   
    
   # def get_sample(self, idx):
    def __getitem__(self, idx):
        temp = self.data_map[idx]
        wav1, sr = torchaudio.load(temp["left_target"])
        wav2, sr = torchaudio.load(temp["right_target"])
        orig, sr = torchaudio.load(temp["original"])
        wav, sr = torch.cat((wav1, wav2), dim=0), sr
        orig = torch.cat((orig, orig), dim=0)
        label = temp['label']

        # Return a dict with target, original, label, sample rate
        # return {
        #     "target": wav,
        #     "original": orig,
        #     "label": label,
        #     "sr": sr
        # }
        return wav, orig, label, sr
