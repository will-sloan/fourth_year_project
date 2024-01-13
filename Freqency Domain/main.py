# Third party shit
print("Started main")
import torch
import os
import torch.nn as nn
import glob
print("Imported")
# Local imports
from SpectroDataset import SpectroDataset
from Autoencoder import AutoEncoder
from SpectroConverter import SpectroConverter
from WavSplitter import WavSplitter

unsplit = True
unprocessed = True
data_folder = '/workspace/extension/unet'
long_mono = 'unchopped/mono.wav'
all_angles = [0, 15, 30, 45, 90, 105, 120, 135, 150, 165, 180]
long_left = 'unchopped/left_'
long_right = 'unchopped/right_'
mono_target = 'wavs/mono'
left_target = 'wavs/left'
right_target = 'wavs/right'
print("Set words")

mono_freq = 'freq/mono'
left_freq = 'freq/left'
right_freq = 'freq/right'

mono_wavs = os.path.join(data_folder, mono_target)
left_wavs = os.path.join(data_folder, left_target)
right_wavs = os.path.join(data_folder, right_target)

print("Made paths")

# Convert wavs into stft
mono_wavs_output_dir = os.path.join(data_folder, mono_freq)
left_wavs_output_dir = os.path.join(data_folder, left_freq)
right_wavs_output_dir = os.path.join(data_folder, right_freq)
print("Created output dirs")
if unsplit:
    print("Start split")
    # Delete all files in mono_wavs
    [os.remove(f) for f in glob.glob(os.path.join(mono_wavs, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(left_wavs, '*')) if os.path.isfile(f)]
    [os.remove(f) for f in glob.glob(os.path.join(right_wavs, '*')) if os.path.isfile(f)]
    print("Cleared Folders")
    # Split data into x sec wav files
    split_length = 5 # seconds
    
    WavSplitter(os.path.join(data_folder, long_mono), mono_wavs, split_length)
    for i in all_angles:
        # Create new filename for the i angle
        long_left = long_left + str(i) + '.wav'
        WavSplitter(os.path.join(data_folder, long_left), left_wavs, split_length)
    print('Left done')
    for i in all_angles:
        # Create new filename for the i angle
        long_right = long_right + str(i) + '.wav'
        WavSplitter(os.path.join(data_folder, long_right), right_wavs, split_length)
    print('Right done')
    
    
if unprocessed:
    print("Start processing")
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

print("All Done")
exit()
# load the dataset
sp = SpectroDataset(mono_wavs_output_dir, left_wavs_output_dir, right_wavs_output_dir, chunk_size=1000, max_chunks=1)
sp.process()

sp.save_data_map('/workspace/extension/fancy_tiny.pkl')
print("Saved data map")


# sp = SpectroDataset(None, None, None)
# sp.load_data_map('/workspace/extension/fancystd.pkl')
# print("Loaded data map")

# exit()
# print(sp[0][1])
# print(sp[0][2])
# print(sp[0][3])

# # Divide the dataset into training, validation, and testing
train_size = int(0.8 * len(sp))
test_size = int(0.1 * len(sp))
val_size = len(sp) - train_size - test_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])

# Load the model
# Real and imaginary parts
num_channels = 2

model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

model.apply(weights_init)
model.train()



# Train the model
model.train_loop(train_dataset, val_dataset, batch_size=32, epochs=100, lr=0.01)

exit()
output = model(sp[0][0].cuda(), sp[0][1])
# Convert back to denormal
real_part = mono[:, :, 0]
imag_part = mono[:, :, 1]
print(real_part.shape)
print(imag_part.shape)

# Denormalize real and imaginary parts
denormalized_real_part = (real_part * sp.std_value.numpy()) + sp.mean_value
denormalized_imag_part = (imag_part * sp.std_value.numpy()) + sp.mean_value
print(denormalized_real_part.shape)
print(denormalized_imag_part.shape)
# Combine denormalized real and imaginary parts
back = torch.stack([denormalized_real_part, denormalized_imag_part], dim=-1)


# Test the model

# save the model

# load a test 
# Convert output to wav
# See how it sounds
# Save the wav

