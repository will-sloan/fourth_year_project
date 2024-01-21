# Third party shit
print("Started main")
import torch
import os
import torch.nn as nn
import glob
from torch.utils.tensorboard import SummaryWriter
print("Imported")
# Local imports
from wav_dataset import WavDataset
from SequenceModel import SequenceModel
from SpectroConverter import SpectroConverter
from WavSplitter import WavSplitter
from SidedSpectroDataset import SidedSpectroDataset

# mono_wav = '/workspace/extension/unet/unchopped2/mono.wav'
import torchaudio

# Load the audio file
# mono_wav, sample_rate = torchaudio.load(mono_wav)

# Get rid of the first 3612 samples
# mono_wav = mono_wav[:, 3612:]

# Save the modified audio to a new file
# torchaudio.save('/workspace/extension/unet/unchopped2/mono_matched.wav', mono_wav, sample_rate)

# mono_wav = '/workspace/extension/unet/unchopped2/mono_matched.wav'

# left_wav_0 = '/workspace/extension/unet/unchopped2/left_0.wav'
# left_wav_15 = '/workspace/extension/unet/unchopped2/left_15.wav'
# left_wav_30 = '/workspace/extension/unet/unchopped2/left_30.wav'
# left_wav_45 = '/workspace/extension/unet/unchopped2/left_45.wav'
# left_wav_60 = '/workspace/extension/unet/unchopped2/left_60.wav'
# left_wav_75 = '/workspace/extension/unet/unchopped2/left_75.wav'
# left_wav_90 = '/workspace/extension/unet/unchopped2/left_90.wav'
# left_wav_105 = '/workspace/extension/unet/unchopped2/left_105.wav'
# left_wav_120 = '/workspace/extension/unet/unchopped2/left_120.wav'
# left_wav_135 = '/workspace/extension/unet/unchopped2/left_135.wav'
# left_wav_150 = '/workspace/extension/unet/unchopped2/left_150.wav'
# left_wav_165 = '/workspace/extension/unet/unchopped2/left_165.wav'
# left_wav_180 = '/workspace/extension/unet/unchopped2/left_180.wav'

# mono_wav, sample_rate = torchaudio.load(mono_wav)
sample_rate = 32000
# 0h23m54.5s - 4h20m24s
# Convert into index
talking_start_index = (30*60) * 32000
talking_end_index = (4*60*60) * 32000
print(talking_start_index, talking_end_index)
# exit()
# print(mono_wav.shape)
# mono_wav = mono_wav[:, talking_start_index:talking_end_index]
# print(mono_wav.shape)

# left_wav_0, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_0.wav')
# print(left_wav_0.shape)
# left_wav_0 = left_wav_0[:, talking_start_index:talking_end_index]
# print(left_wav_0.shape)

# left_wav_15, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_15.wav')
# print(left_wav_15.shape)
# left_wav_15 = left_wav_15[:, talking_start_index:talking_end_index]
# print(left_wav_15.shape)

# left_wav_30, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_30.wav')
# print(left_wav_30.shape)
# left_wav_30 = left_wav_30[:, talking_start_index:talking_end_index]
# print(left_wav_30.shape)

# left_wav_45, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_45.wav')
# print(left_wav_45.shape)
# left_wav_45 = left_wav_45[:, talking_start_index:talking_end_index]
# print(left_wav_45.shape)

# left_wav_60, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_60.wav')
# print(left_wav_60.shape)
# left_wav_60 = left_wav_60[:, talking_start_index:talking_end_index]
# print(left_wav_60.shape)

# left_wav_75, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_75.wav')
# print(left_wav_75.shape)
# left_wav_75 = left_wav_75[:, talking_start_index:talking_end_index]
# print(left_wav_75.shape)

# left_wav_90, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_90.wav')
# print(left_wav_90.shape)
# left_wav_90 = left_wav_90[:, talking_start_index:talking_end_index]
# print(left_wav_90.shape)

# left_wav_105, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_105.wav')
# print(left_wav_105.shape)
# left_wav_105 = left_wav_105[:, talking_start_index:talking_end_index]
# print(left_wav_105.shape)

# left_wav_120, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_120.wav')
# print(left_wav_120.shape)
# left_wav_120 = left_wav_120[:, talking_start_index:talking_end_index]
# print(left_wav_120.shape)

# left_wav_135, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_135.wav')
# print(left_wav_135.shape)
# left_wav_135 = left_wav_135[:, talking_start_index:talking_end_index]
# print(left_wav_135.shape)

# left_wav_150, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_150.wav')
# print(left_wav_150.shape)
# left_wav_150 = left_wav_150[:, talking_start_index:talking_end_index]
# print(left_wav_150.shape)

# left_wav_165, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_165.wav')
# print(left_wav_165.shape)
# left_wav_165 = left_wav_165[:, talking_start_index:talking_end_index]
# print(left_wav_165.shape)

# left_wav_180, _ = torchaudio.load('/workspace/extension/unet/unchopped2/left_180.wav')
# print(left_wav_180.shape)
# left_wav_180 = left_wav_180[:, talking_start_index:talking_end_index]
# print(left_wav_180.shape)


# # Save all the files
# torchaudio.save('/workspace/extension/unet/chopped/mono_matched.wav', mono_wav, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_0_matched.wav', left_wav_0, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_15_matched.wav', left_wav_15, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_30_matched.wav', left_wav_30, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_45_matched.wav', left_wav_45, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_60_matched.wav', left_wav_60, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_75_matched.wav', left_wav_75, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_90_matched.wav', left_wav_90, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_105_matched.wav', left_wav_105, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_120_matched.wav', left_wav_120, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_135_matched.wav', left_wav_135, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_150_matched.wav', left_wav_150, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_165_matched.wav', left_wav_165, sample_rate)
# torchaudio.save('/workspace/extension/unet/chopped/left_180_matched.wav', left_wav_180, sample_rate)

# Do the same for the right
right_wav_0, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_0.wav')
print(right_wav_0.shape)
right_wav_0 = right_wav_0[:, talking_start_index:talking_end_index]
print(right_wav_0.shape)

right_wav_15, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_15.wav')
print(right_wav_15.shape)
right_wav_15 = right_wav_15[:, talking_start_index:talking_end_index]
print(right_wav_15.shape)

right_wav_30, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_30.wav')
print(right_wav_30.shape)
right_wav_30 = right_wav_30[:, talking_start_index:talking_end_index]
print(right_wav_30.shape)

right_wav_45, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_45.wav')
print(right_wav_45.shape)
right_wav_45 = right_wav_45[:, talking_start_index:talking_end_index]
print(right_wav_45.shape)

right_wav_60, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_60.wav')
print(right_wav_60.shape)
right_wav_60 = right_wav_60[:, talking_start_index:talking_end_index]
print(right_wav_60.shape)

right_wav_75, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_75.wav')
print(right_wav_75.shape)
right_wav_75 = right_wav_75[:, talking_start_index:talking_end_index]
print(right_wav_75.shape)

right_wav_90, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_90.wav')
print(right_wav_90.shape)
right_wav_90 = right_wav_90[:, talking_start_index:talking_end_index]
print(right_wav_90.shape)

right_wav_105, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_105.wav')
print(right_wav_105.shape)
right_wav_105 = right_wav_105[:, talking_start_index:talking_end_index]
print(right_wav_105.shape)

right_wav_120, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_120.wav')
print(right_wav_120.shape)
right_wav_120 = right_wav_120[:, talking_start_index:talking_end_index]
print(right_wav_120.shape)

right_wav_135, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_135.wav')
print(right_wav_135.shape)
right_wav_135 = right_wav_135[:, talking_start_index:talking_end_index]
print(right_wav_135.shape)

right_wav_150, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_150.wav')
print(right_wav_150.shape)
right_wav_150 = right_wav_150[:, talking_start_index:talking_end_index]
print(right_wav_150.shape)

right_wav_165, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_165.wav')
print(right_wav_165.shape)
right_wav_165 = right_wav_165[:, talking_start_index:talking_end_index]
print(right_wav_165.shape)

right_wav_180, _ = torchaudio.load('/workspace/extension/unet/unchopped2/right_180.wav')
print(right_wav_180.shape)
right_wav_180 = right_wav_180[:, talking_start_index:talking_end_index]
print(right_wav_180.shape)

# Save the files
torchaudio.save('/workspace/extension/unet/chopped/right_0_matched.wav', right_wav_0, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_15_matched.wav', right_wav_15, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_30_matched.wav', right_wav_30, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_45_matched.wav', right_wav_45, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_60_matched.wav', right_wav_60, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_75_matched.wav', right_wav_75, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_90_matched.wav', right_wav_90, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_105_matched.wav', right_wav_105, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_120_matched.wav', right_wav_120, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_135_matched.wav', right_wav_135, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_150_matched.wav', right_wav_150, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_165_matched.wav', right_wav_165, sample_rate)
torchaudio.save('/workspace/extension/unet/chopped/right_180_matched.wav', right_wav_180, sample_rate)




exit()
left_wav_0 = '/workspace/extension/unet/unchopped2/left_0.wav'
left_wav_15 = '/workspace/extension/unet/unchopped2/left_15.wav'
left_wav_30 = '/workspace/extension/unet/unchopped2/left_30.wav'
left_wav_45 = '/workspace/extension/unet/unchopped2/left_45.wav'
left_wav_60 = '/workspace/extension/unet/unchopped2/left_60.wav'
left_wav_75 = '/workspace/extension/unet/unchopped2/left_75.wav'
left_wav_90 = '/workspace/extension/unet/unchopped2/left_90.wav'
left_wav_105 = '/workspace/extension/unet/unchopped2/left_105.wav'
left_wav_120 = '/workspace/extension/unet/unchopped2/left_120.wav'
left_wav_135 = '/workspace/extension/unet/unchopped2/left_135.wav'
left_wav_150 = '/workspace/extension/unet/unchopped2/left_150.wav'
left_wav_165 = '/workspace/extension/unet/unchopped2/left_165.wav'
left_wav_180 = '/workspace/extension/unet/unchopped2/left_180.wav'

right_wav_0 = '/workspace/extension/unet/unchopped2/right_0.wav'
right_wav_15 = '/workspace/extension/unet/unchopped2/right_15.wav'
right_wav_30 = '/workspace/extension/unet/unchopped2/right_30.wav'
right_wav_45 = '/workspace/extension/unet/unchopped2/right_45.wav'
right_wav_60 = '/workspace/extension/unet/unchopped2/right_60.wav'
right_wav_75 = '/workspace/extension/unet/unchopped2/right_75.wav'
right_wav_90 = '/workspace/extension/unet/unchopped2/right_90.wav'
right_wav_105 = '/workspace/extension/unet/unchopped2/right_105.wav'
right_wav_120 = '/workspace/extension/unet/unchopped2/right_120.wav'
right_wav_135 = '/workspace/extension/unet/unchopped2/right_135.wav'
right_wav_150 = '/workspace/extension/unet/unchopped2/right_150.wav'
right_wav_165 = '/workspace/extension/unet/unchopped2/right_165.wav'
right_wav_180 = '/workspace/extension/unet/unchopped2/right_180.wav'

exit()

sp = WavDataset(left_wav_0, mono_wav, 0, 10)
sp.save_data_map("/workspace/extension/unet/left_0.pkl")

sp = WavDataset(left_wav_15, mono_wav, 15, 100)
sp.save_data_map("/workspace/extension/unet/left_15.pkl")

sp = WavDataset(left_wav_30, mono_wav, 30, 100)
sp.save_data_map("/workspace/extension/unet/left_30.pkl")

sp = WavDataset(left_wav_45, mono_wav, 45, 100)
sp.save_data_map("/workspace/extension/unet/left_45.pkl")

sp = WavDataset(left_wav_60, mono_wav, 60, 100)
sp.save_data_map("/workspace/extension/unet/left_60.pkl")

sp = WavDataset(left_wav_75, mono_wav, 75, 100)
sp.save_data_map("/workspace/extension/unet/left_75.pkl")

sp = WavDataset(left_wav_90, mono_wav, 90, 100)
sp.save_data_map("/workspace/extension/unet/left_90.pkl")

sp = WavDataset(left_wav_105, mono_wav, 105, 100)
sp.save_data_map("/workspace/extension/unet/left_105.pkl")

sp = WavDataset(left_wav_120, mono_wav, 120, 100)
sp.save_data_map("/workspace/extension/unet/left_120.pkl")

sp = WavDataset(left_wav_135, mono_wav, 135, 100)
sp.save_data_map("/workspace/extension/unet/left_135.pkl")

sp = WavDataset(left_wav_150, mono_wav, 150, 100)
sp.save_data_map("/workspace/extension/unet/left_150.pkl")

sp = WavDataset(left_wav_165, mono_wav, 165, 100)
sp.save_data_map("/workspace/extension/unet/left_165.pkl")

sp = WavDataset(left_wav_180, mono_wav, 180, 100)
sp.save_data_map("/workspace/extension/unet/left_180.pkl")

sp = WavDataset(right_wav_0, mono_wav, 0, 100)
sp.save_data_map("/workspace/extension/unet/right_0.pkl")

sp = WavDataset(right_wav_15, mono_wav, 15, 100)
sp.save_data_map("/workspace/extension/unet/right_15.pkl")

sp = WavDataset(right_wav_30, mono_wav, 30, 100)
sp.save_data_map("/workspace/extension/unet/right_30.pkl")

sp = WavDataset(right_wav_45, mono_wav, 45, 100)
sp.save_data_map("/workspace/extension/unet/right_45.pkl")

sp = WavDataset(right_wav_60, mono_wav, 60, 100)
sp.save_data_map("/workspace/extension/unet/right_60.pkl")

sp = WavDataset(right_wav_75, mono_wav, 75, 100)
sp.save_data_map("/workspace/extension/unet/right_75.pkl")

sp = WavDataset(right_wav_90, mono_wav, 90, 100)
sp.save_data_map("/workspace/extension/unet/right_90.pkl")

sp = WavDataset(right_wav_105, mono_wav, 105, 100)
sp.save_data_map("/workspace/extension/unet/right_105.pkl")

sp = WavDataset(right_wav_120, mono_wav, 120, 100)
sp.save_data_map("/workspace/extension/unet/right_120.pkl")

sp = WavDataset(right_wav_135, mono_wav, 135, 100)
sp.save_data_map("/workspace/extension/unet/right_135.pkl")

sp = WavDataset(right_wav_150, mono_wav, 150, 100)
sp.save_data_map("/workspace/extension/unet/right_150.pkl")

sp = WavDataset(right_wav_165, mono_wav, 165, 100)
sp.save_data_map("/workspace/extension/unet/right_165.pkl")

sp = WavDataset(right_wav_180, mono_wav, 180, 100)
sp.save_data_map("/workspace/extension/unet/right_180.pkl")
# model = SequenceModel().cuda()
# model.train()
# print("Starting treain loop")
# model.train_loop(sp, 32, 4)


