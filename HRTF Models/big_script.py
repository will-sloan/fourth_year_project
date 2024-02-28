# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, encoding_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.LeakyReLU(True),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(True),
#             nn.Linear(256, encoding_dim),
#             nn.Tanh()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, 256),
#             nn.LeakyReLU(True),
#             nn.Linear(256, 256),
#             nn.LeakyReLU(True),
#             nn.Linear(256, input_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

import torch
import torch.nn as nn

class Unflatten(nn.Module):
    def __init__(self, num_channels, seq_length):
        super(Unflatten, self).__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, self.num_channels, self.seq_length)

class Conv1DAutoencoder(nn.Module):
    def __init__(self, channels, encoding_dim):
        super(Conv1DAutoencoder, self).__init__()
        # Assuming the input size is [batch_size, channels, seq_length] = [32, 2, 512]
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3, stride=2, padding=1),  # Adjusted for correct output size
            nn.LeakyReLU(True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # Adjusted for correct output size
            nn.LeakyReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 128, encoding_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32 * 128),
            nn.LeakyReLU(True),
            Unflatten(32, 128),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted for correct output size
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(16, channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Adjusted for correct output size
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # print(f"Input shape: {x.shape}")
        # # return x
        # for i, layer in enumerate(self.encoder):
        #     x = layer(x)
        #     print(f"After encoder layer {i+1} ({layer.__class__.__name__}): {x.shape}")
        
        # # for i, layer in enumerate(self.decoder):
        # #     x = layer(x)
        # #     print(f"After decoder layer {i+1} ({layer.__class__.__name__}): {x.shape}")
        
        return x



class HRIRTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, n_angle_bins, angle_embedding_dim):
        super(HRIRTransformer, self).__init__()
        self.autoencoder = Conv1DAutoencoder(2, encoding_dim)
        self.positional_encoder = PositionalEncoding(d_model=encoding_dim)  # Adjust for angle embedding

        self.angle_embedding = nn.Embedding(n_angle_bins, angle_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_dim, nhead=nhead, dim_feedforward=dim_feedforward)  # Adjust for angle embedding
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=encoding_dim, nhead=nhead, dim_feedforward=dim_feedforward)  # Adjust for angle embedding
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(encoding_dim, output_dim)  # Adjust for angle embedding
    
    def forward(self, src, src_angle_idx, tgt, tgt_angle_idx, diffs):
        '''
        src: (N, S, E) - (batch size, sequence length, input dimension) - (32, 2, 512)
        tgt: (N, T, E) - (batch size, sequence length, input dimension) - (32, 1, 512)
        src_angle_idx: (N, ) - (batch size, ) - (32, )
        tgt_angle_idx: (N, ) - (batch size, ) - (32, )
        diffs: (N, S, E) - (batch size, sequence length, input dimension) - (32, 2, 512)
        diff_angles: (N, ) - (batch size, ) - (32, )
        '''
        # Convert angles to embeddings
        # print(src_angle_idx)
        # print(tgt_angle_idx)
        # print(diff_angles)
        src_angle_embedding = self.angle_embedding(src_angle_idx)
        tgt_angle_embedding = self.angle_embedding(tgt_angle_idx)

        # print("Src_angle_embedding: ", src_angle_embedding.shape)
        # print("Tgt_angle_embedding: ", tgt_angle_embedding.shape)
        # diffs_angle_embedding = self.angle_embedding(diff_angles)

        # src_angle_embedding: (N, E) - (batch size, angle embedding dimension) - (32, 256)
        # tgt_angle_embedding: (N, E) - (batch size, angle embedding dimension) - (32, 256)
        # diffs_angle_embedding: (N, E) - (batch size, angle embedding dimension) - (32, 256)

        # Encode src and tgt using autoencoder's encoder
        # print("Before encoding src: ", src.shape)
        
        src_encoded = self.autoencoder.encoder(src)
        # src_encoded = self.autoencoder(src)
        # print("Before encoding tgt: ", tgt.shape)
        tgt_encoded = self.autoencoder.encoder(tgt)
        # print("Before encoding diffs: ", diffs.shape)
        diffs_encoded = self.autoencoder.encoder(diffs)
        # print("Src_encoded: ", src_encoded.shape)
        # print("Tgt_encoded: ", tgt_encoded.shape)
        # print("Diffs_encoded: ", diffs_encoded.shape)

        # src_encoded: (N, E) - (batch size, encoding dimension) - (32, 256)
        # tgt_encoded: (N, E) - (batch size, encoding dimension) - (32, 256)
        # diffs_encoded: (N, E) - (batch size, encoding dimension) - (32, 256)

        # Convert angle embeddings to (32, 1, 16)
        src_angle_embedding = src_angle_embedding.unsqueeze(1)
        tgt_angle_embedding = tgt_angle_embedding.unsqueeze(1)
        # diffs_angle_embedding = diffs_angle_embedding.unsqueeze(0)

        # src_angle_embedding: (N, 1, E) - (batch size, 1, angle embedding dimension) - (32, 1, 256)
        # tgt_angle_embedding: (N, 1, E) - (batch size, 1, angle embedding dimension) - (32, 1, 256)

        

        # add dimension to encoded - (32, 1, 256)
        src_encoded = src_encoded.unsqueeze(1)
        tgt_encoded = tgt_encoded.unsqueeze(1)
        diffs_encoded = diffs_encoded.unsqueeze(1)

        # src_encoded: (N, 1, E) - (batch size, 1, encoding dimension) - (32, 1, 256)
        # tgt_encoded: (N, 1, E) - (batch size, 1, encoding dimension) - (32, 1, 256)
        # diffs_encoded: (N, 1, E) - (batch size, 1, encoding dimension) - (32, 1, 256)

        # Combine encoded HRIR with angle embeddings
        # src_combined: add src + src_angle_embedding + diffs_encoded + diffs_angle_embedding + tgt_angle_embedding - (32, 4, 256)
        src_combined = torch.cat((src_encoded, src_angle_embedding, diffs_encoded, tgt_angle_embedding), dim=1)
        # print("Src_combined: ", src_combined.shape)

        # Apply positional encoding
        src_pos_encoded = self.positional_encoder(src_combined)
        tgt_pos_encoded = self.positional_encoder(tgt_encoded)
        # print("Src_pos_encoded: ", src_pos_encoded.shape)
        # print("Tgt_pos_encoded: ", tgt_pos_encoded.shape)

        # src_pos_encoded: (S, N, E) - (sequence length, batch size, encoding dimension) - (32, 4, 256)
        # tgt_pos_encoded: (T, N, E) - (sequence length, batch size, encoding dimension) - (32, 1, 256)

        # Swap to (S, N, E) - (sequence length, batch size, encoding dimension)
        src_pos_encoded = src_pos_encoded.permute(1, 0, 2)
        tgt_pos_encoded = tgt_pos_encoded.permute(1, 0, 2)

        # src_pos_encoded: (S, N, E) - (sequence length, batch size, encoding dimension) - (4, 32, 256)
        # tgt_pos_encoded: (T, N, E) - (sequence length, batch size, encoding dimension) - (1, 32, 256)

        memory = self.transformer_encoder(src_pos_encoded)

        # memory: (S, N, E) - (sequence length, batch size, encoding dimension) - (4, 32, 256)

        output = self.transformer_decoder(tgt_pos_encoded, memory)

        # output: (T, N, E) - (sequence length, batch size, encoding dimension) - (1, 32, 256)

        # Remove the first dimension
        output = output.squeeze(0)

        output = self.fc_out(output)

        # output: (N, E) - (batch size, output dimension) - (32, 512)

        # add dimension to output - (32, 1, 512)
        output = output.unsqueeze(1)       

        return output
    

def angle_to_index(angle, n_bins=72):
    # Assuming angle is given in degrees and ranges from 0 to 359
    bin_size = 360 / n_bins
    index = (angle // bin_size).type(torch.long)
    return index

input_dim = 512
output_dim = 512
encoding_dim = 256
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
n_angle_bins = 72
angle_embedding_dim = 256




import sys
sys.path.append('/workspace/fourth_year_project/HRTF Models/')

from HRIRDataset import HRIRDataset
# from BasicTransformer import BasicTransformer

sofa_file = '/workspace/fourth_year_project/HRTF Models/sofa_hrtfs/RIEC_hrir_subject_001.sofa'
# Basic Dataset only loads the HRIRs at 0 degrees and 90 degrees for baseline and 45 degree for testing
hrir_dataset = HRIRDataset()
for i in range(1,100):
    hrir_dataset.load(sofa_file.replace('001', str(i).zfill(3)))

import numpy as np
from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import torch.nn as nn

device = torch.device('cuda')

loss_function = nn.MSELoss()

num_epochs = 200 # Number of epochs to train for

# total_epochs = 100
warmup_epochs = 10
base_lr = 1e-3

# Define lambda function for the learning rate schedule
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # Linear warm-up
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        # Exponential decay
        decay_rate = 0.95  # Decay rate
        decay_epochs = epoch - warmup_epochs  # Subtract warmup epochs
        return pow(decay_rate, decay_epochs)


train_size = int(0.7 * len(hrir_dataset))
val_size = int(0.2 * len(hrir_dataset))
test_size = len(hrir_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(hrir_dataset, [train_size, val_size, test_size])

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)

model = HRIRTransformer(input_dim, output_dim, encoding_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, n_angle_bins, angle_embedding_dim)

model = model.to(device)  # Move model to the specified device


model.train()
total_loss = 0
# test_val = None
# import time

optimizer = optim.Adam(model.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for hrir0, angle0, hrir_target, target_angle in train_loader:
    # move all to device
    hrir0, angle0, hrir_target, target_angle = hrir0.to(device), angle0.to(device), hrir_target.to(device), target_angle.to(device)
    #convert hrirs to floats and angles to ints
    hrir0 = hrir0.float()
    hrir_target = hrir_target.float()
    angle0 = angle0.long()
    target_angle = target_angle.long()
    angle0 = angle_to_index(angle0)
    target_angle = angle_to_index(target_angle)
    # time.sleep(5)

    # normalize between -1 and 1
    # hrir0 = (hrir0 - hrir0.mean()) / hrir0.std()
    # hrir90 = (hrir90 - hrir90.mean()) / hrir90.std()
    # hrir45 = (hrir45 - hrir45.mean()) / hrir45.std()
    diffs = hrir_target - hrir0
    # diff_angles = target_angle - angle0
    output = model(hrir0, angle0, hrir_target, target_angle, diffs)
    # print(test_val.shape)
    real_target = hrir_target[:,0,:].unsqueeze(1)


    loss = loss_function(output, real_target)
    
    # # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

print(total_loss / len(train_loader))

# Update the learning rate
# scheduler.step()

# return total_loss / len(data_loader)