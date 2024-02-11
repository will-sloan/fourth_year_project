import torch
from torch import nn
import time
class BackupWorkingModel(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=12):
        super(BackupWorkingModel, self).__init__()
        # [seq_length, batch_size, d_model] = [512, n, 3]
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
    
    def autoregressive_inference(self, input):
        # [seq_length, batch_size, d_model]
        output = torch.zeros_like(input)
        # Set first token to be the same as the input
        output[0, :] = input[0, :]
        # print(output.shape, input.shape)
        with torch.no_grad():
            for t in range(1, output.size(0)):
                # Get the predictions for the next token
                predictions = self.transformer(input, output)
                
                # Save the predicted values for the next token
                output[t, :] = predictions[t, :]

                # print(t, output.shape)
        
        return output
    
    def forward(self, src, angle, tgt=None, src_mask=None, tgt_mask=None):
        if tgt is not None:
            # create a 512 long angle tensor to add as a feature
            angle = angle.view(src.shape[0], 1, 1)
            angle = angle.expand(-1, -1, src.shape[2])
            src = torch.cat((src, angle), dim=1)
            # [batch_size, 2, 512] --> [batch_size, 3, 512]
            # Do the same for tgt but set it to -1 since that isn't a real angle
            constant_tgt = torch.full_like(angle, -1)
            tgt = torch.cat((tgt, constant_tgt), dim=1)
            src = src.permute(2, 0, 1)
            tgt = tgt.permute(2, 0, 1)
            # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
            # Replace the first value of tgt with the first value of src
            # Since when we generate, we won't have a value for tgt
            tgt[0,:] = src[0,:]
            
            output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
            # Convert back to [batch_size, d_model, seq_length]
            output = output.permute(1, 2, 0)
        else:
            # No value for tgt so we generate one
            # print("no valu")
            angle = angle.view(src.shape[0], 1, 1)
            angle = angle.expand(-1, -1, src.shape[2])
            # print(angle.shape, src.shape)
            src = torch.cat((src, angle), dim=1)
            # print(src.shape)
            # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
            src = src.permute(2, 0, 1)
            output = self.autoregressive_inference(src)
            # Converted back to [batch_size, d_model, seq_length]

        return output