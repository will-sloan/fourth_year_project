# Learns using teacher forcing

import torch
from torch import nn
import time
import random
import torch.utils.checkpoint as checkpoint

class MaskAutoregressive(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=12):
        super(MaskAutoregressive, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.linear = nn.Linear(d_model, d_model-1)
    
    def autoregressive_inference(self, input):
        # [seq_length, batch_size, d_model]
        output = torch.zeros_like(input)
        # Set first token to be the same as the input
        output[0, :] = input[0, :]
        # print(output.shape, input.shape)
        with torch.no_grad():
            for t in range(1, output.size(0)):
                predictions = self.transformer(input, output)
                predictions = self.linear(predictions)
                # [seq_length, batch_size, d_model]
                output[t, :] = predictions[t, :]

        return output
    
    def teacher_force(self, input, tgt, teacher_ratio=0.5):
        output = torch.zeros_like(input)
        # turn off grad
        output.requires_grad = False
        # Set first token to be the same as the input
        output[0, :] = input[0, :]

        for t in range(input.size(0)):
            predictions = checkpoint.checkpoint(self.transformer, input, output, use_reentrant=False)
            predictions = self.linear(predictions)
            random_number = random.random()
            use_teacher_forcing = random_number < teacher_ratio
            # [seq_length, batch_size, d_model
            if use_teacher_forcing:
                # Replaces the sequence value in all batches to be the tgt
                print(output.shape, tgt.shape, t)
                output[t, :] = tgt[t, :]
            else:
                print(output.shape, tgt.shape, t)
                output[t, :] = predictions[t, :]
        output = output.detach().clone()
        output.requires_grad = True
        return output
    
    def forward(self, src, angle, tgt=None, epoch=None):
        # create a 512 long angle tensor to add as a feature
        angle = angle.view(src.shape[0], 1, 1)
        angle = angle.expand(-1, -1, src.shape[2])
        src = torch.cat((src, angle), dim=1)

        # tgt is going to be the unmasked version of the src
        if tgt is not None:
            # tgt and src are already the same
            src = src.permute(2, 0, 1)
            tgt = tgt.permute(2, 0, 1)
            output = self.teacher_force(input=src, tgt=tgt)
            output = output.permute(1, 2, 0)
        else:
            # No value for tgt so we predict one token at a time
            # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
            src = src.permute(2, 0, 1)
            output = self.autoregressive_inference(src)
            # Converted back to [batch_size, d_model, seq_length]
            output = output.permute(1, 2, 0)

        return output

