import torch
from torch import nn

class MainModel(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=8):
        super(MainModel, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
    
    def autoregressive_inference(self, input, angle):
        # Initialize the output sequence with a start-of-sequence token
        output = torch.zeros_like(input)
        output[:, 0] = input[:, 0]

        # Loop over each position in the output sequence
        for t in range(1, output.size(1)):
            # Get the predictions for the next token
            predictions = self.transformer(input, output, angle)
            
            # Choose the token with the highest prediction as the next token
            # This is a simple "greedy" decoding method; other methods like beam search could also be used
            output[:, t] = predictions[:, t-1].argmax(dim=-1)
        
        return output
    
    def forward(self, src, tgt, angle, src_mask=None, tgt_mask=None):
        if self.training and tgt is not None:
            # create a 512 long angle tensor to add as a feature
            angle = angle.view(src.shape[0], 1, 1)
            angle = angle.expand(-1, -1, src.shape[2])
            src = torch.cat((src, angle), dim=1)
            # Do the same for tgt but set it to -1 since that isn't a real angle
            constant_tgt = torch.full_like(angle, -1)
            tgt = torch.cat((tgt, constant_tgt), dim=1)
            
            # Replace the first value of tgt with the first value of src
            # Since when we generate, we won't have a value for tgt
            tgt = torch.cat([src[:, :1], tgt[:, 1:]], dim=1)
            src = src.permute(2, 0, 1)
            tgt = tgt.permute(2, 0, 1)
            
            output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            # Convert back to [batch_size, d_model, seq_length]
            output = output.permute(1, 2, 0)
        else:
            # No value for tgt so we generate one
            output = self.autoregressive_inference(src, angle)
        return output