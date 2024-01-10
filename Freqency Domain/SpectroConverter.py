import os
import librosa
import numpy as np


class SpectroConverter():

    def __init__(self, wavs, output_dir):
        # Convert mono wavs
        file_list = os.listdir(wavs)

        # Loop through all files
        for file in file_list:
            # For each file, load in as wave then convert to spectrogram
            audio_path = os.path.join(wavs, file)
            y, sr = librosa.load(audio_path, sr=32000)
            # Define the sample rate, window length, hop length, and FFT size
            sr = 32000  # Sample rate in Hz (change this to match your audio's sample rate)
            window_length_ms = 25  # Window length in ms
            hop_length_ms = 10  # Hop length in ms

            # Convert window length and hop length from ms to samples
            window_length_samples = int(sr * window_length_ms / 1000)
            hop_length_samples = int(sr * hop_length_ms / 1000)

            n_fft = window_length_samples
            # Compute the STFT with a Hanning window
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length_samples, win_length=window_length_samples, window='hann')

            #D = D[:, 0]
            #D = librosa.stft(y, n_fft=1102, window='hann', )
            # Save the spectrogram in output
            # left_90_442.npy
            np.save(os.path.join(output_dir, file.replace('.wav', '')), D)
