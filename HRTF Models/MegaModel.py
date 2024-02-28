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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, encoding_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class HRIRTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, encoding_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, n_angle_bins, angle_embedding_dim):
        super(HRIRTransformer, self).__init__()
        self.autoencoder = Autoencoder(input_dim, encoding_dim)
        self.positional_encoder = PositionalEncoding(d_model=encoding_dim + angle_embedding_dim)  # Adjust for angle embedding

        self.angle_embedding = nn.Embedding(n_angle_bins, angle_embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_dim + angle_embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)  # Adjust for angle embedding
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=encoding_dim + angle_embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)  # Adjust for angle embedding
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(encoding_dim + angle_embedding_dim, output_dim)  # Adjust for angle embedding

    def forward(self, src, tgt, src_angle_idx, tgt_angle_idx):
        # Convert angles to embeddings
        src_angle_embedding = self.angle_embedding(src_angle_idx)
        tgt_angle_embedding = self.angle_embedding(tgt_angle_idx)

        # Encode src and tgt using autoencoder's encoder
        src_encoded = self.autoencoder.encoder(src)
        tgt_encoded = self.autoencoder.encoder(tgt)

        # Combine encoded HRIR with angle embeddings
        src_combined = torch.cat((src_encoded, src_angle_embedding), dim=1)
        tgt_combined = torch.cat((tgt_encoded, tgt_angle_embedding), dim=1)

        # Apply positional encoding
        src_pos_encoded = self.positional_encoder(src_combined)
        tgt_pos_encoded = self.positional_encoder(tgt_combined)

        memory = self.transformer_encoder(src_pos_encoded)
        output = self.transformer_decoder(tgt_pos_encoded, memory)
        output = self.fc_out(output)
        return output
    

def angle_to_index(angle, n_bins=72):
    # Assuming angle is given in degrees and ranges from 0 to 359
    bin_size = 360 / n_bins
    index = int(angle // bin_size)
    return index

input_dim = 512
output_dim = 512
encoding_dim = 128
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 512
n_angle_bins = 72
angle_embedding_dim = 16


