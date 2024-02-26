import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
import torch.nn.functional as F
class MyTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.5, sos_token=0, eos_token=-3):
        super(MyTransformer, self).__init__()
        self.attention_weights = nn.Linear(d_model, 1)

        self.sos_token = sos_token
        self.eos_token = eos_token

        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder setup
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder setup
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        # Adjusting the model to output a single HRIR channel
        # Assuming d_model is the dimensionality you want for the HRIR (e.g., 512)
        self.final_projection = nn.Linear(d_model, d_model)
        self.reduce_to_single_channel = nn.Linear(nhead * num_decoder_layers, 1)  # Adjust for your architecture specifics
        self.d_model = d_model
    
    def forward(self, src_with_angle, target_sequence=None):
        # print(src_with_angle.shape) # torch.Size([32, 3, 192])
        # Assuming src_with_angle is a tensor of shape [batch, seq_len, features]
        src_with_angle = src_with_angle.permute(1, 0, 2)  # Shape: [seq_len, batch, features]
        src_with_angle = self.pos_encoder(src_with_angle)
        memory = self.transformer_encoder(src_with_angle)
        
        if target_sequence is not None:
            # If a target sequence is provided, use it as the input to the decoder
            # Ensure it's permuted correctly to [seq_len, batch, features] if not already
            target_sequence = target_sequence.permute(1, 0, 2)
            target_sequence = self.pos_encoder(target_sequence)
            decoder_input = target_sequence
        else:
            # If no target is provided, create a dummy sequence for decoder input
            dummy_sequence = torch.full(src_with_angle.size(), fill_value=self.sos_token, device=src_with_angle.device)
            dummy_sequence = self.pos_encoder(dummy_sequence)
            decoder_input = dummy_sequence
        
        # Process through the decoder
        output = self.transformer_decoder(decoder_input, memory)
        output = output.permute(1, 0, 2)
        # Compute attention scores
        attn_scores = self.attention_weights(output).squeeze(-1)  # Shape: [batch, seq_len]
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # Shape: [batch, seq_len, 1]
        
        # # Apply attention weights
        output_attended = torch.sum(output * attn_weights, dim=1, keepdim=True)  # Shape: [batch, 1, features]
        # output_attended = output_attended.permute() # Shape: [batch, 1, features]
        # print(output_attended.shape)
        
        return output_attended
        # print(output.shape)
        # return output
