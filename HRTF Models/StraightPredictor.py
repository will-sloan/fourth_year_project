import torch
from torch import nn
import time
import random
import torch.utils.checkpoint as checkpoint

class StraightPredictor(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=16):
        super(StraightPredictor, self).__init__()
        # [seq_length, batch_size, d_model] = [512, n, 3]
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.num_iter = 3

    def forward(self, src, angle, tgt=None):
        # create a 512 long angle tensor to add as a feature
        angle = angle.view(src.shape[0], 1, 1)
        angle = angle.expand(-1, -1, src.shape[2])
        src = torch.cat((src, angle), dim=1)
        # [batch_size, 2, 512] --> [batch_size, 3, 512]
        # Do the same for tgt but set it to -1 since that isn't a real angle
        if tgt is not None:
            constant_tgt = torch.full_like(angle, -1)
            tgt = torch.cat((tgt, constant_tgt), dim=1)
        else:
            tgt = torch.full_like(src, -2)
        src = src.permute(2, 0, 1)
        tgt = tgt.permute(2, 0, 1)
        # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
        # Replace the first value of tgt with the first value of src
        # Since when we generate, we won't have a value for tgt
        tgt[0,:] = src[0,:]
        
        output = self.transformer(src, tgt)
        output = output.permute(1, 2, 0)
        return output
    

