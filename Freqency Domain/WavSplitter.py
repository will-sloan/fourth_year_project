# takes a the directory where 3 files are:
# left.wav --> 5 hours of left ear audio
# right.wav --> 5 hours of right ear audio
# mono.wav --> 5 hours of mono audio
import os
import torchaudio
import torch

class WavSplitter():

    def __init__(self, input_dir, output_dir, angle):
        self.directory = input_dir
        self.output_dir = output_dir
        # Split the wav files into 5 second chunks
        self.split_wav_files()

    def split_wav_files(self, split_length=5):
        # Split the mono wav file
        self.split_wav_file(os.path.join(self.directory, f'mono_{angle}.wav'), split_length)
        # Split the left wav file
        self.split_wav_file(os.path.join(self.directory, f'left_{angle}.wav'), split_length)
        # Split the right wav file
        self.split_wav_file(os.path.join(self.directory, f'right_{angle}.wav'), split_length)

    def split_wav_file(self, wav_file, split_length):
        w, s = torchaudio.load(wav_file)
        chunk_length = split_length * s
        num_chunks = w.shape[1] // chunk_length
        for i in range(num_chunks):
            # Get start index
            start_idx = i * chunk_length
            # Get end index
            end_idx = min((i + 1) * chunk_length, w.shape[1])

            # Get the chunk
            chunk = w[:, start_idx:end_idx]

            # Pad the chunk if it is too short
            if chunk.shape[1] < chunk_length:
                pad = torch.zeros((2, chunk_length - chunk.shape[1]))
                # stereo audio so we need to pad 2 channels
                chunk = torch.cat((chunk, pad), dim=1)

            # Save the chunk and a copy of the label file
            new_file_name = f"{wav_file.replace('.wav', '')}_{self.angle}_{i}.wav"
            new_file_path = os.path.join(self.output_dir, new_file_name)
            torchaudio.save(new_file_path, chunk, s)