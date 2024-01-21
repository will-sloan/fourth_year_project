# Third party shit
print("Started main")
import torch
import os
import torch.nn as nn
import glob
from torch.utils.tensorboard import SummaryWriter
print("Imported")
# Local imports
from SpectroDataset import SpectroDataset
from Autoencoder import AutoEncoder
from SpectroConverter import SpectroConverter
from WavSplitter import WavSplitter
from SidedSpectroDataset import SidedSpectroDataset

unsplit = False
unprocessed = False
data_folder = '/workspace/extension/unet'
long_mono = 'unchopped2/mono.wav'
all_angles = [0, 15, 30, 45, 90, 105, 120, 135, 150, 165, 180]
long_left = 'unchopped2/left_'
long_right = 'unchopped2/right_'
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
        temp = long_left + str(i) + '.wav'
        print("Splitting " + temp)
        WavSplitter(os.path.join(data_folder, temp), left_wavs, split_length)
    print('Left done')
    for i in all_angles:
        # Create new filename for the i angle
        temp = long_right + str(i) + '.wav'
        print("Splitting " + temp)
        WavSplitter(os.path.join(data_folder, temp), right_wavs, split_length)
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


# load the dataset
#sp = SpectroDataset(mono_wavs_output_dir, left_wavs_output_dir, right_wavs_output_dir, chunk_size=1000, max_chunks=1)
print(mono_wavs_output_dir)
print(left_wavs_output_dir)
sp = SidedSpectroDataset(mono_wavs_output_dir, left_wavs_output_dir, chunk_size=240)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
import gc
num_channels = 2
# Train left_model
left_model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()
left_model.apply(weights_init)
left_model.train()

# Load model from checkpoint
chkpfile = '/workspace/extension/unet/left_model_checkpoints/model_loopnum_0_4_batch_30.pt'
chkp = torch.load(chkpfile)
left_model.load_state_dict(chkp['model_state_dict'])
left_model.optimizer.load_state_dict(chkp['optimizer_state_dict'])
# for i in range(8):
left_model.train_loop(sp, batch_size=8, epochs=8, writer=None, loop_num=0, name='left_model_checkpoints1')

# Clear left_model from memory
del left_model
torch.cuda.empty_cache()  # Clear unused memory from GPU
gc.collect()  # Clear unused memory from CPU

sp = SidedSpectroDataset(mono_wavs_output_dir, right_wavs_output_dir, chunk_size=240)

# Train right_model
right_model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()
right_model.apply(weights_init)
right_model.train()
# for i in range(8):
right_model.train_loop(sp, batch_size=8, epochs=8, writer=None, loop_num=0, name='right_model_checkpoints')

# Gets 1000 samples
# sp.load_chunk()

#sp.save_data_map('/workspace/extension/fancy_tiny.pkl')
#print("Saved data map")


# sp = SpectroDataset(None, None, None)
# sp.load_data_map('/workspace/extension/fancystd.pkl')
# print("Loaded data map")

# exit()


# # Divide the dataset into training, validation, and testing
# train_size = int(0.8 * len(sp))
# test_size = int(0.1 * len(sp))
# val_size = len(sp) - train_size - test_size
#train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])

# Load the model
# Real and imaginary parts
# num_channels = 2

# left_model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()



# left_model.apply(weights_init)
# left_model.train()


# # Train the model
# #model.train_loop(train_dataset, val_dataset, batch_size=32, epochs=100, lr=0.01)
# # Pass writer
# # writer = SummaryWriter('runs/mixed_run')
# for i in range(10):
#     left_model.train_loop(sp, batch_size=8, epochs=8, writer=None,loop_num=i, name='right_model_checkpoints')
# Close writer
# writer.close()
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

