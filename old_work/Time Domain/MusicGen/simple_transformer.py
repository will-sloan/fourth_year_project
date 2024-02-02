# A simple transformer that can't be run until we get more memory

import torch
from torch import nn
import torch.nn.functional as F

class AudioTransformer(nn.Module):
    def __init__(self, comp_model, d_model, nhead, num_layers, dim_feedforward, compute_seperate_loss=False, input_size=500):
        super(AudioTransformer, self).__init__()
        self.input_encoding = nn.Linear(input_size, d_model)  # input audio
        self.input_bn = nn.BatchNorm1d(d_model)

        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.hidden_layer = nn.Linear(d_model, d_model)
        self.output_bn = nn.BatchNorm1d(d_model)
        self.output_decoding = nn.Linear(d_model, input_size)  # Decoding back to stereo audio
        #self.angle_encoding = nn.Linear(6, d_model)  # add this once 90 works. 
        
        if comp_model is not None:
            self.comp_model = comp_model.cuda()
        else:
            self.comp_model = None
        self.compute_seperate_loss = compute_seperate_loss

    # Orig and target are the normalized values of the codes
    def forward(self, orig, target, angle):
        orig = orig.cuda()
        target = target.cuda()
        #print(audio.shape)
        orig = self.input_encoding(orig)
        target = self.input_encoding(target)
        # Relu to rid negatives
        #orig = F.relu(orig)
        #target = F.relu(target)

        #angle = self.angle_encoding(angle)  # Process one-hot encoded angle
        #angle = angle.unsqueeze(1).repeat(1, audio.size(2), 1)  # Repeat angle for each time step
        #x = audio + angle  # Combine audio and angle

        x = self.transformer(src=orig, tgt=target)
        #x = F.relu(x)

        x = self.output_decoding(x)
        x = F.relu(x)

        # Scale back to integers
        #x = x * 1000
        #x = torch.round(x)

        return x

    def compress(self, stereo):
            if self.comp_model is None:
                raise Exception("No compression model found")
            stereo = stereo.cuda()
            with torch.no_grad():
                stereo, scale = self.comp_model.encode(stereo)
            return stereo


    def decompress(self, stereo):
            if self.comp_model is None:
                raise Exception("No compression model found")
            stereo = stereo.cuda()
            with torch.no_grad():
                stereo = self.comp_model.decode(stereo)
            return stereo
    
    def compute_mean_std(self):
        all_data = torch.cat([item['target'] for item in self.data_map] + [item['original'] for item in self.data_map])
        mean = torch.mean(all_data)
        std = torch.std(all_data)
        return mean, std

    
    
    def train_loop(self, dataset, batch_size=1, epochs=1, lr=0.001, cosine_loss=False):
        if not cosine_loss:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CosineEmbeddingLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        

        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            #for i, (target, target_norm, orig, orig_norm, angle, sr) in enumerate(train_loader):
            for i, (_, target, _, orig, angle, sr) in enumerate(train_loader):
                optimizer.zero_grad()
                target = target.cuda()
                orig = orig.cuda()
                # Convert wav to codes
                target_codes = target
                orig_codes = orig
                # Pass codes to model
                output = self(orig=orig_codes.float(), target=target_codes.float(), angle=angle)
                #output = output.squeeze(0)
                #target = target_codes.squeeze(0)
                
                # Using MSE loss
                output = output.float()
                target_codes = target_codes.float()
                loss = loss_fn(output, target_codes)

                #print(output, target_codes)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if i % 10 == 0:
                    print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")
                if i == 0:
                    print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")

            if epoch % 50 == 0 and epoch != 0:
                torch.save(self.state_dict(), f"model_{epoch}.pth")
                print(f"Saved model_{epoch}.pth")

        print("Finished Training")