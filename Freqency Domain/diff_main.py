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
run_model = True
data_folder = '/workspace/extension/unet'
long_mono = 'chopped/mono.wav'
all_angles = [0, 15, 30, 45, 90, 105, 120, 135, 150, 165, 180]
long_left = 'chopped/left_'
long_right = 'chopped/right_'
mono_target = 'wavs/mono'
left_target = 'wavs/left'
right_target = 'wavs/right'
print("Set words")

mono_freq = 'norm_freq/mono'
left_freq = 'norm_freq/left'
right_freq = 'norm_freq/right'

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
    print("done splitting mono")
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
    # SpectroConverter(mono_wavs=mono_wavs, mono_output_dir=mono_wavs_output_dir, left_wavs=left_wavs, left_output_dir=left_wavs_output_dir, right_wavs=right_wavs, right_output_dir=right_wavs_output_dir)
    SpectroConverter(wavs=mono_wavs, output_dir=mono_wavs_output_dir)
    # print('Mono done')
    SpectroConverter(wavs=left_wavs, output_dir=left_wavs_output_dir)
    # print('Left done')
    SpectroConverter(wavs=right_wavs, output_dir=right_wavs_output_dir)
    # print('Right done')

if run_model:
    sp = SpectroDataset(mono_wavs_output_dir, left_wavs_output_dir, chunk_size=1920)

    def weights_init(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
    import gc
    num_channels = 2
    # Train left_model
    left_model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()
    left_model.apply(weights_init) 

    # Load from checkpoint
    chkppath = '/workspace/extension/unet/left_model_checkpoints5/model_loopnum_5_50_part2.pt'
    left_model.load_state_dict(torch.load(chkppath)["model_state_dict"])
    left_model.train()
    for i in range(8):
        left_model.train_loop(sp, batch_size=8, epochs=100, writer=None, loop_num=i, name='left_model_checkpoints5', bonus='part3')

    # Clear left_model from memory
    del left_model
    torch.cuda.empty_cache()  # Clear unused memory from GPU
    gc.collect()  # Clear unused memory from CPU


    sp = SpectroDataset(mono_wavs_output_dir, right_wavs_output_dir, chunk_size=1920)
    # Train right_model
    right_model = AutoEncoder(input_channels=num_channels, out_channels=num_channels).cuda()
    right_model.apply(weights_init)
    # Load from checkpoint
    # chkppath = '/workspace/extension/unet/right_model_checkpoints3/model_loopnum_0_7_batch_18.pt'
    # right_model.load_state_dict(torch.load(chkppath)["model_state_dict"])  
    right_model.train()
    for i in range(8):
        right_model.train_loop(sp, batch_size=8, epochs=100, writer=None, loop_num=i, name='right_model_checkpoints5', bonus='part3')

