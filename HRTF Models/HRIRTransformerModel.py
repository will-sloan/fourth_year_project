import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding

class HRIRTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(HRIRTransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.angle_encoder = nn.Linear(1, d_model)  # Encodes the angle
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(d_model, 2)  # Assuming the output HRIR has 2 channels per time-point
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.angle_encoder.bias.data.zero_()
        self.angle_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, angle):
        src = self.pos_encoder(src)
        angle_encoded = self.angle_encoder(angle).unsqueeze(0).repeat(src.size(0), 1, 1)  # Repeat angle encoding for each time point
        src = src + angle_encoded
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
