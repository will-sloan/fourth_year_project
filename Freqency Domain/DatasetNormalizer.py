import os
import numpy as np
import torch

class DatasetNormalizer():
    def __init__(self, input_dirs, output_dir):
        self.input_dirs = input_dirs
        self.output_dir = output_dir
        #                                                                        left         left_105_2000.npy
        self.file_list = self.file_list = [os.path.join(dir, file) for dir in input_dirs for file in os.listdir(dir)]
        #                                                                                                   left         left_105_2000.npy
        self.output_file_list = [os.path.join(self.output_dir, os.path.relpath(filepath, dir)) for dir in self.input_dirs for filepath in self.file_list]
        self.global_min_real = np.inf
        self.global_max_real = -np.inf
        self.global_min_imag = np.inf
        self.global_max_imag = -np.inf

    def compute_global_min_max(self):
        '''
        -159.55472
        160.03091
        -158.56412
        160.77977
        '''
        # for filepath in self.file_list:
        #     data = np.load(filepath)
        #     real_data = data.real
        #     imag_data = data.imag
        #     self.global_min_real = min(self.global_min_real, np.min(real_data))
        #     self.global_max_real = max(self.global_max_real, np.max(real_data))
        #     self.global_min_imag = min(self.global_min_imag, np.min(imag_data))
        #     self.global_max_imag = max(self.global_max_imag, np.max(imag_data))

        # self.global_min_imag = -159.55472
        # self.global_max_imag = 160.03091
        # self.global_min_real = -158.56412
        # self.global_max_real = 160.77977

    def normalize(self, data):
        normalized_real = (data.real - self.global_min_real) / (self.global_max_real - self.global_min_real)
        normalized_imag = (data.imag - self.global_min_imag) / (self.global_max_imag - self.global_min_imag)
        return normalized_real + 1j * normalized_imag

    def process_file(self, input_filepath, output_filepath):
        # Load the data
        data = np.load(input_filepath)
        # Remove the extra row and column
        data = data[:-1, :-1]

        # Normalize the data
        normalized_data = self.normalize(data)

        # add real and imag channels
        normalized_data = torch.view_as_real(torch.from_numpy(normalized_data))
        # permute to have channels first
        normalized_data = normalized_data.permute(2, 0, 1)

        # Save the normalized data
        torch.save(normalized_data, output_filepath)

    def process_all_files(self):
        print("processing all")
        self.compute_global_min_max()
        print("computed global min max")
        for input_filepath, output_filepath in zip(self.file_list, self.output_file_list):
                self.process_file(input_filepath, output_filepath)