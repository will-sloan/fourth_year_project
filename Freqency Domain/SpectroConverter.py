import os
import librosa
import numpy as np
import torch

class SpectroConverter():

    def __init__(self, wavs, output_dir):
        # Convert mono wavs
        file_list = os.listdir(wavs)

        # Loop through all files
        for file in file_list:
            # For each file, load in as wave then convert to spectrogram
            audio_path = os.path.join(wavs, file)
            y, sr = librosa.load(audio_path, sr=32000)
            #normalize the audio signal
            y = self.normalize(y)
            spec = librosa.stft(y, n_fft=512, hop_length=160, win_length=400, center=True)

            # Save the spectrogram in output
            # left_90_442.npy
            real = np.expand_dims(np.real(spec), axis=0)
            imag = np.expand_dims(np.imag(spec), axis=0)
            spec = np.concatenate((real, imag), axis=0)
            spec = torch.from_numpy(spec)
            #np.save(os.path.join(output_dir, file.replace('.wav', '')), D)
            torch.FloatTensor(spec).save(os.path.join(output_dir, file.replace('.wav', '')))

    def normalize(self, samples, desired_rms = 0.1, eps = 1e-4):
        rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
        samples = samples * (desired_rms / rms)
        return samples