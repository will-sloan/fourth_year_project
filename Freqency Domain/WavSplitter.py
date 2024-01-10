# takes a the directory where 3 files are:
# left.wav --> 5 hours of left ear audio
# right.wav --> 5 hours of right ear audio
# mono.wav --> 5 hours of mono audio
import os
import torchaudio
import torch

class WavSplitter():

    def __init__(self, input_file, output_dir, split_length=5):
        self.filename = input_file.split('/')[-1]
        self.output_dir = output_dir
        '''
        Splitting /workspace/extension/unet/unchopped/mono.wav into 5 second chunks
        Outputting to /workspace/extension/unet/wavs/mono
        '''
        print(f'Splitting {self.filename} into {split_length} second chunks')
        print(f'Outputting to {self.output_dir}')
        # Split the wav files into 5 second chunks
        assert self.filename.endswith('.wav'), 'File must be a wav file'

        w, s = torchaudio.load(input_file)
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

            # Save the chunk
                
            new_file_name = f"{self.filename.replace('.wav', '')}_{i}.wav"
            new_file_path = os.path.join(self.output_dir, new_file_name)
            torchaudio.save(new_file_path, chunk, s)