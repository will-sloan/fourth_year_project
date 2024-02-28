import math
import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
import torch.nn.functional as F
class BasicTransformer(nn.Module):
    # d_model=192, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, angle_dim=64
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.5, max_len=512):
        super(BasicTransformer, self).__init__()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.angle_encoder = nn.Linear(1, d_model)  # Encodes angle to a high-dimensional space
        self.encoder = nn.Linear(d_model, d_model)  # Adjusted for concatenated angles
        self.decoder = nn.Linear(d_model, d_model)  # Output HRIR prediction
        self.d_model = d_model
        self.angle_dim = d_model
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.angle_encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, hrir0, angle0, hrir90, angle90, hrir45, angle45):
        # batch, seq_len, d_model = 32, n, 512
        # Encode the angles
        angle0 = self.angle_encoder(angle0)
        angle90 = self.angle_encoder(angle90)
        angle45 = self.angle_encoder(angle45)

        return 0

        # Concatenate the encoded angles with the HRIR data
        # angles = [32, 1, 512]
        # hrir0 = [32, 2, 512]
        # concated = [32, 3, 512]
        concated0 = torch.cat([hrir0, angle0], dim=1)
        concated90 = torch.cat([hrir90, angle90], dim=1)
        # concated45 = torch.cat([hrir45, angle45], dim=1)

        # Concate the 3 angles
        # concated = [32, 9, 512]
        concated = torch.cat([concated0, concated90, angle45], dim=1)

        # Encode the concatenated data
        src = self.encoder(concated) * math.sqrt(self.d_model)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Encode the data
        memory = self.transformer_encoder(src)

        # Decode the data
        # Encode the target data
        tgt_encoded = self.encoder(hrir45) * math.sqrt(self.d_model)  # Apply encoding and scale
        tgt_pos_encoded = self.pos_encoder(tgt_encoded)  # Apply positional encoding

        # Decode the data
        output = self.transformer_decoder(tgt_pos_encoded, memory)
        output = self.decoder(output)
        return output

