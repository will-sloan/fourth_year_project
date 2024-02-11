import torch
from torch import nn
import time
import random
import torch.utils.checkpoint as checkpoint

class AutoregressiveModel(nn.Module):
    def __init__(self, d_model=3, nhead=3, num_layers=12):
        super(AutoregressiveModel, self).__init__()
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
                predictions = self.transformer(input, output)
                output[t, :] = predictions[t, :]

        return output
    
    def teacher_force(self, input, tgt=None, teacher_ratio=0.5, epoch=None):
        # teacher_ratio is probability of using tgt instead of the predicted value
        # The higher the value, the more likely we are to use tgt
        # Opposite to the other mtehods
        if epoch is not None:
            teacher_ratio = min(teacher_ratio - 0.05 * epoch, 0.1)

        # output = torch.zeros_like(input)
        # output[0, :] = input[0, :]

        # intermediate_outputs = [input[0, :]]
        
        output = torch.zeros_like(input)
        # turn off grad
        output.requires_grad = False
        # Set first token to be the same as the input
        output[0, :] = input[0, :]

        # for t in range(input.size(0)):
        # limit to 250 tokens
        for t in range(min(input.size(0), 250)):
            predictions = checkpoint.checkpoint(self.transformer, input, output, use_reentrant=False)

            random_number = random.random()
            use_teacher_forcing = random_number < teacher_ratio
            # [seq_length, batch_size, d_model
            if use_teacher_forcing and tgt is not None:
                # Replaces the sequence value in all batches to be the tgt
                output[t, :] = tgt[t, :]
            else:
                output[t, :] = predictions[t, :]
        # remove first value
        # intermediate_outputs.pop(0)
        output[250:, :] = tgt[250:, :]
        # output = torch.stack(intermediate_outputs, dim=0)
        output = output.detach().clone()
        output.requires_grad = True
        return output
    

    # Decrease tgt_ratio to decrease amount of times the method is run
    def forward_old(self, src, angle, tgt=None, tgt_ratio=0.5, epoch=None):
        # create a 512 long angle tensor to add as a feature
        angle = angle.view(src.shape[0], 1, 1)
        angle = angle.expand(-1, -1, src.shape[2])
        src = torch.cat((src, angle), dim=1)
        # Generate a random number between 0 and 1
        random_number = random.random()
        if epoch is not None:
            # increase tgt_ratio to increase amount of times the method is run
            tgt_ratio = min(tgt_ratio + 0.01 * epoch, 0.5)
        # If the random number is less than the teacher forcing ratio, use teacher forcing method
        if random_number < tgt_ratio:
            use_teacher_forcing = True
        else:
            use_teacher_forcing = False
        if tgt is not None:
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

            # Only use teacher forcing some of the time,
            # Also, within teacher forcing, we only use tgt some of the time
            if use_teacher_forcing:
                print(f"Using teacher forcing, {tgt_ratio} epoch {epoch}")
                output = self.teacher_force(input=src, tgt=tgt, teacher_ratio=0.8, epoch=epoch)
            else:
                if self.training:
                    # print("Using checkpoint")
                    output = checkpoint.checkpoint(self.transformer, src, tgt, use_reentrant=False)
                else:
                    output = self.transformer(src, tgt)
            # Convert back to [batch_size, d_model, seq_length]
            output = output.permute(1, 2, 0)
        else:
            # No value for tgt so we predict one token at a time
            # [batch_size, d_model, seq_length] --> [seq_length, batch_size, d_model]
            src = src.permute(2, 0, 1)
            output = self.autoregressive_inference(src)
            # Converted back to [batch_size, d_model, seq_length]
            output = output.permute(1, 2, 0)

        return output
    
    def forward(self, src, angle, tgt):
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
        
        output = self.transformer(src, tgt)
        output = output.permute(1, 2, 0)
        return output
    

