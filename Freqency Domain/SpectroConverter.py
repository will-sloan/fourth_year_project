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
            # y = self.normalize(y)
            spec = librosa.stft(y, n_fft=1024, hop_length=160, win_length=400, center=True)

            # Save the spectrogram in output
            # left_90_442.npy
            real = np.expand_dims(np.real(spec), axis=0)
            imag = np.expand_dims(np.imag(spec), axis=0)
            spec = np.concatenate((real, imag), axis=0)
            spec = torch.from_numpy(spec)
            #np.save(os.path.join(output_dir, file.replace('.wav', '')), D)
            torch.save(spec, os.path.join(output_dir, file.replace('.wav', '')))

    # def normalize(self, samples, desired_rms = 0.1, eps = 1e-4):
    #     rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    #     samples = samples * (desired_rms / rms)
    #     return samples
            

'''
import os
import librosa
import numpy as np
import torch

class SpectroConverter():

    def __init__(self, mono_wavs, mono_output_dir, left_wavs, left_output_dir, right_wavs, right_output_dir):
        # Convert mono wavs
        file_list = os.listdir(left_wavs)

        # Loop through all files
        for file in file_list:
            # For each file, load in as wave then convert to spectrogram
            angle = file.split('_')[0]
            index = file.split('_')[1].replace('.wav', '')
            # left_0_0.wav
            left_audio_path = os.path.join(left_wavs, file)
            # right_0_0.wav
            right_audio_path = os.path.join(right_wavs, f'right_{angle}_{index}.wav')
            # mono_0.wav
            mono_audio_path = os.path.join(mono_wavs, f'mono_{index}.wav')

            left_wav, _ = librosa.load(left_audio_path, sr=32000)
            right_wav, _ = librosa.load(right_audio_path, sr=32000)
            mono_wav, sr = librosa.load(mono_audio_path, sr=32000)
            
            # Compute the matched filter between left and mono
            # left_filter = np.convolve(left_wav[:32000], mono_wav[:32000])
            # time_shift = np.argmax(left_filter)
            # print(time_shift)

            # right_filter = np.convolve(right_wav[:32000], mono_wav[:32000])
            # time_shift = np.argmax(right_filter)
            # print(time_shift)
            # continue
            spec = librosa.stft(y, n_fft=512, hop_length=160, win_length=400, center=True)

            # Save the spectrogram in output
            # left_90_442.npy
            real = np.expand_dims(np.real(spec), axis=0)
            imag = np.expand_dims(np.imag(spec), axis=0)
            spec = np.concatenate((real, imag), axis=0)
            spec = torch.from_numpy(spec)
            #np.save(os.path.join(output_dir, file.replace('.wav', '')), D)
            torch.save(spec, os.path.join(output_dir, file.replace('.wav', '')))

    # def normalize(self, samples, desired_rms = 0.1, eps = 1e-4):
    #     rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    #     samples = samples * (desired_rms / rms)
    #     return samples
'''