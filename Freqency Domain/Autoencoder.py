import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



class AutoEncoder(nn.Module):
    def __init__(self, input_channels=2, out_channels=2, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )   

        self.encoder = nn.Sequential(
            conv_block(input_channels, 64),
            nn.MaxPool2d(2), 
            conv_block(64, 128),
            nn.MaxPool2d(2), 
        )

        self.angle_to_index = {0.0: 0, 45.0: 1, 90.0: 2, 135.0: 3}
        # Add one channel for the angle
        self.bottleneck = conv_block(128 + len(self.angle_to_index.keys()), 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), 
            conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            conv_block(64, 64),
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        

    def forward(self, x, angle):
        #x = torch.view_as_real(x)
        # Before this: batch, fft, time, real/imag
        #x = x.permute(0, 3, 1, 2)
        # after this: batch, real/imag, fft, time
        # x = x[:,:,:-1,:-1]
        #print("In: ", x.shape)
        # Break down encoder into its 4 steps
        e1 = self.encoder[0](x)
        #print("e1: ", e1.shape)
        e2 = self.encoder[1](e1)
        #print("e2: ", e2.shape)
        e3 = self.encoder[2](e2)
        #print("e3: ", e3.shape)
        e4 = self.encoder[3](e3)
        #print("e4: ", e4.shape)
        #x = self.encoder(x)
        #print("En: ", x.shape)
        
        # Pass the result through the bottleneck
        angle_tensor = torch.zeros((len(angle), 4)).to(x.device)
        for i, a in enumerate(angle):
            angle_tensor[i, self.angle_to_index[a]] = 1
        # Expand the angle tensor to match the shape of the encoder output
        angle_tensor = angle_tensor.unsqueeze(2).unsqueeze(3)
        angle_tensor = angle_tensor.expand(-1, -1, e4.shape[2], e4.shape[3])
        e4 = torch.cat([e4, angle_tensor], dim=1)
        #print("e4: ", e4.shape)
        x = self.bottleneck(e4)
        #print("bt: ", x.shape)

        # Pass the result through the decoder
        # Break down decoder into its 7 steps
        d1 = self.decoder[0](x)
        #print("d1: ", d1.shape)
        d2 = self.decoder[1](d1)
        #print("d2: ", d2.shape)
        d3 = self.decoder[2](d2)
        #print("d3: ", d3.shape)
        d4 = self.decoder[3](d3)
        #print("d4: ", d4.shape)
        # d5 = self.decoder[4](d4)
        # print("d5: ", d5.shape)

        # d6 = self.decoder[5](d5)
        #x = self.decoder(d5)
        # print("de: ", d6.shape)
        


        # Pass the result through the final convolution
        x = self.final_conv(d4)
        # Return to original shape
        x = x.permute(0, 2, 3, 1)
        #print("Out: ", x.shape)
        #x = torch.view_as_complex(x)
        return x


    def train_loop(self, train_dataset, val_dataset, batch_size, epochs, lr):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        #optimizer = optim.Adam(self.parameters(), lr=lr)
        # Add momentum to the optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()
        #print("Starting training loop")

        for epoch in range(epochs):
            #print("Epoch", epoch)
            for i, (mono, label, right, left) in enumerate(dataloader):
                inputs, targets = mono.cuda(), right.cuda()
                #inputs = inputs[:, :-1, :-1]
                #targets = targets[:, :-1, :-1]

                #label = tuple(float(l) for l in label)
                
                #inputs = torch.view_as_real(inputs)
                #print("Target shape: ", targets.shape)
                #print("Input shape: ", inputs.shape)
                optimizer.zero_grad()
                # Convert to real
                
                #print(inputs.shape)
                outputs = self(inputs, label)
                #print("Output shape: ", outputs.shape)
                outputs = outputs.contiguous()
                
                # Convert back to complex
                outputs = torch.view_as_complex(outputs)
                inputs = torch.view_as_complex(inputs)
                #print("Bring back complex: ", outputs.shape)
                # The output from the model is a mask that represents the difference between the input and the target
                # We can use the difference to get both left and right channels
                
                #print("Input stft shape: ", inputs.shape)
                output_stft = inputs * outputs
                #print("Output stft shape: ", output_stft.shape)

                # Compare this to the target stft

                # We compare the istft to the target
                # XD --> output stft
                # targets --> stft
                # loss = criterion(output_stft, targets)
                real_loss = criterion(output_stft.real, targets.real)
                imag_loss = criterion(output_stft.imag, targets.imag)
                loss = real_loss + imag_loss
                #print("Loss: ", loss.item())
                loss.backward()
                optimizer.step()
                #exit()

             # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                for i, (mono, label, right, left) in enumerate(val_loader):
                    inputs, targets = mono.cuda(), right.cuda()
                    # inputs = inputs[:, :-1, :-1]
                    # targets = targets[:, :-1, :-1]
                    # label = tuple(float(l) for l in label)
                    # inputs = torch.view_as_real(inputs)

                    outputs = self(inputs, label)
                    
                    outputs = outputs.contiguous()
                    outputs = torch.view_as_complex(outputs)
                    inputs = torch.view_as_complex(inputs)
                    output_stft = inputs * outputs

                    val_loss = val_loss + criterion(output_stft.real, targets.real).item() + criterion(output_stft.imag, targets.imag).item()
                val_loss /= len(val_loader)  # Calculate average validation loss

            scheduler.step()
            if optimizer.param_groups[0]['lr'] < 0.0001:
                optimizer.param_groups[0]['lr'] = 0.0001
            print(f'Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}')

            # After every 10 epochs, save the model checkpoint
            if epoch % 10 == 0:
                torch.save(self.state_dict(), f'/workspace/extension/unet/model_checkpoints/model_std_{epoch}.pt')
            