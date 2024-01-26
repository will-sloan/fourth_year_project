import os
import torch

# Directory containing the spectrograms
dirs = ['/workspace/extension/unet/norm_freq/mono', '/workspace/extension/unet/norm_freq/left', '/workspace/extension/unet/norm_freq/right']

# Get a list of all files in the directory
for dir_path in dirs:
    print("Start ", dir_path)
    file_names = os.listdir(dir_path)

    for file_name in file_names:
        # Construct the full file path
        file_path = os.path.join(dir_path, file_name)

        # Load the spectrogram
        spectrogram = torch.load(file_path)

        # Normalize the real and imaginary parts separately
        real_normalized = (spectrogram[0] - spectrogram[0].mean()) / spectrogram[0].std()
        imag_normalized = (spectrogram[1] - spectrogram[1].mean()) / spectrogram[1].std()

        # Combine the normalized real and imaginary parts
        spectrogram_normalized = torch.stack((real_normalized, imag_normalized), dim=0)

        # Save the normalized spectrogram
        torch.save(spectrogram_normalized, file_path)
    print("End ", dir_path)