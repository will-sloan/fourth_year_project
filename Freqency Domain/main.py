# Third party shit
import torch
import os
import torch.nn as nn
import glob

# Local imports
from SpectroDataset import SpectroDataset
from Autoencoder import AutoEncoder
from SpectroConverter import SpectroConverter
from WavSplitter import WavSplitter

unsplit = False
unprocessed = False
data_folder = '/workspace/extension/unet'
long_mono = 'unchopped/mono.wav'
long_left = 'unchopped/left_90.wav'
long_right = 'unchopped/right_90.wav'
mono_target = 'wavs/mono'
left_target = 'wavs/left'
right_target = 'wavs/right'

mono_freq = 'freq/mono'
left_freq = 'freq/left'
right_freq = 'freq/right'

mono_wavs = os.path.join(data_folder, mono_target)
left_wavs = os.path.join(data_folder, left_target)
right_wavs = os.path.join(data_folder, right_target)

# Convert wavs into stft
mono_wavs_output_dir = os.path.join(data_folder, mono_freq)
left_wavs_output_dir = os.path.join(data_folder, left_freq)
right_wavs_output_dir = os.path.join(data_folder, right_freq)

if unsplit:
    # Delete all files in mono_wavs
    [os.remove(f) for f in glob.glob(os.path.join(mono_wavs, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(left_wavs, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(right_wavs, '*')) if os.path.isfile(f)]
    print("Cleared Folders")
    # Split data into x sec wav files
    split_length = 5 # seconds
    
    WavSplitter(os.path.join(data_folder, long_mono), mono_wavs, split_length)
    WavSplitter(os.path.join(data_folder, long_left), left_wavs, split_length)
    WavSplitter(os.path.join(data_folder, long_right), right_wavs, split_length)
    print('Split done')
    
    
if unprocessed:
    # Delete all files in output dirs
    [os.remove(f) for f in glob.glob(os.path.join(mono_wavs_output_dir, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(left_wavs_output_dir, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(right_wavs_output_dir, '*')) if os.path.isfile(f)]

    # Set the wav files to train on

    # Run conversion
    SpectroConverter(wavs=mono_wavs, output_dir=mono_wavs_output_dir)
    print('Mono done')
    SpectroConverter(wavs=left_wavs, output_dir=left_wavs_output_dir)
    print('Left done')
    SpectroConverter(wavs=right_wavs, output_dir=right_wavs_output_dir)
    print('Right done')

# load the dataset
sp = SpectroDataset(mono_wavs_output_dir, left_wavs_output_dir, right_wavs_output_dir)
sp.process()

sp.save_data_map('/workspace/extension/fancy.pkl')
print("Saved data map")
    
#sp = SpectroDataset(None, None, None)
#sp.load_data_map('/workspace/extension/fancy.pkl')

# print(sp[0][0])
# print(sp[0][1])
# print(sp[0][2])
# print(sp[0][3])

# # Divide the dataset into training, validation, and testing
train_size = int(0.8 * len(sp))
test_size = int(0.1 * len(sp))
val_size = len(sp) - train_size - test_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])

# Load the model
model = AutoEncoder()

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

model.apply(weights_init)
model.train()



# Train the model
model.train_loop(train_dataset, val_dataset, batch_size=32, epochs=100, lr=0.01)

# Test the model

# save the model

# load a test 
# Convert output to wav
# See how it sounds
# Save the wav

