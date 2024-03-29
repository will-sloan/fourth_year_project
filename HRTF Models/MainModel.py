import torch
from torch import nn
import time
import random

class MainModel(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=12):
        super(MainModel, self).__init__()
        # [seq_length, batch_size, d_model] = [512, n, 3]
        self.transformer = nn.Transformer(d_model, nhead, num_layers)

    
    def forward(self, src, angle, tgt=None, epoch=None, tgt_ratio=0.5):
        # create a 512 long angle tensor to add as a feature
        backup_shape = src.shape
        angle = angle.view(src.shape[0], 1, 1)
        angle = angle.expand(-1, -1, src.shape[2])
        src = torch.cat((src, angle), dim=1)
        
        # If epoch is provided, calculate tgt_ratio based on epoch
        if epoch is not None:
            # Start with a tgt_ratio of 0.5 and decrease it by 0.01 every epoch
            # However, ensure that tgt_ratio never goes below 0.1
            tgt_ratio = tgt_ratio - 0.01 * epoch
            tgt_ratio = max(tgt_ratio, 0.1)

        random_number = random.random()
        # If the random number is greater than tgt_ratio, we use a empty tgt
        if random_number > tgt_ratio:
            tgt = torch.zeros(backup_shape)
            tgt = tgt.to(src.device)

        # [batch_size, 2, 512] --> [batch_size, 3, 512]
        # Do the same for tgt but set it to -1 since that isn't a real angle
        constant_tgt = torch.full_like(angle, -1)
        tgt = torch.cat((tgt, constant_tgt), dim=1)
        src = src.permute(2, 0, 1)
        tgt = tgt.permute(2, 0, 1)
        tgt[0,:] = src[0,:] 
        # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
        # Replace the first value of tgt with the first value of src
        # Since when we generate, we won't have a value for tgt
        
        output = self.transformer(src, tgt)
        # Convert back to [batch_size, d_model, seq_length]
        output = output.permute(1, 2, 0)
        
        return output