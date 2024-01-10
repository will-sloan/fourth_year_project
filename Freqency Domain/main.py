# Third party shit
import torch
import os

# Local imports
from SpectroDatset import SpectroDataset
from Autoencoder import AutoEncoder
from SpectroConverter import SpectroConverter
from WavSplitter import WavSplitter

unprocessed = True
data_folder = '/workspace/extension/unet'
if unprocessed:
    # Split data into x sec wav files
    mono_wavs = os.path.join(data_folder, 'wavs/mono')
    WavSplitter(mono_wavs)

    # Set the wav files to train on
    left_wavs = os.path.join(data_folder, 'wavs/left')
    right_wavs = os.path.join(data_folder, 'wavs/right')

    # Convert wavs into stft
    mono_wavs_output_dir = os.path.join(data_folder, 'freq/mono')
    left_wavs_output_dir = os.path.join(data_folder, 'freq/left')
    right_wavs_output_dir = os.path.join(data_folder, 'freq/right')
    # Run conversion
    SpectroConverter(mono_wavs=mono_wavs, 
                     left_wavs=left_wavs, 
                     right_wavs=right_wavs, 
                     mono_wavs_output_dir=mono_wavs_output_dir, 
                     left_wavs_output_dir=left_wavs_output_dir,
                     right_wavs_output_dir=right_wavs_output_dir)

# load the dataset
sp = SpectroDataset(data_folder)

# Divide the dataset into training, validation, and testing
train_size = int(0.8 * len(sp))
test_size = int(0.1 * len(sp))
val_size = len(sp) - train_size - test_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])

# Load the model
model = AutoEncoder()

# Train the model

# Test the model

# save the model

# load a test 
# Convert output to wav
# See how it sounds
# Save the wav

