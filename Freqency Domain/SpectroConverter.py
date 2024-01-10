import os
import librosa
import numpy as np


class SpectroConverter():

    def __init__(self, mono_wavs, left_wavs, right_wavs, mono_wavs_output_dir, left_wavs_output_dir, right_wavs_output_dir):
        # Convert mono wavs
        self.convert_wavs(mono_wavs, mono_wavs_output_dir)
        # Convert left wavs
        self.convert_wavs(left_wavs, left_wavs_output_dir)
        # Convert right wavs
        self.convert_wavs(right_wavs, right_wavs_output_dir)


    def convert_wavs(self, target_dir, output_dir):
        file_list = os.listdir(target_dir)

        # Loop through all files
        for file in file_list:
            # For each file, load in as wave then convert to spectrogram
            audio_path = os.path.join(target_dir, file)
            y, sr = librosa.load(audio_path, sr=32000)
            # Define the sample rate, window length, hop length, and FFT size
            sr = 32000  # Sample rate in Hz (change this to match your audio's sample rate)
            n_fft = len(y)
            window_length_samples = len(y)
            hop_length_samples = len(y)
            # Compute the STFT with a Hanning window
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length_samples, win_length=window_length_samples, window='hann')

            D = D[:, 0]
            #D = librosa.stft(y, n_fft=1102, window='hann', )
            # Save the spectrogram in output
            np.save(os.path.join(output_dir, file.replace('.wav', '')), D)
