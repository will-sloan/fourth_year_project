import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



class AutoEncoder(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )   

        self.encoder = nn.Sequential(
            conv_block(552, 128),
            nn.MaxPool2d(2,1),  # 276 x 50
            conv_block(128, 128),
            nn.MaxPool2d(2,1),  # 138 x 25
        )

        self.bottleneck = conv_block(128, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1),  # 276 x 50
            conv_block(128, 128),
            nn.ConvTranspose2d(128, 256, kernel_size=1, stride=1),  # 552 x 100
            conv_block(256, 256),
            nn.ConvTranspose2d(256, 552, kernel_size=1, stride=1),  # 552 x 100
            #nn.ConstantPad2d((0, 1, 0, 0), 0),  
        )

        self.final_conv = nn.Conv2d(552, 552, kernel_size=1)

    def forward(self, x, angle):
        # Pass the input through the encoder
        print("In: ", x.shape)
        x = self.encoder(x)
        print("En: ", x.shape)

        # Pass the result through the bottleneck
        x = self.bottleneck(x)
        print("bt: ", x.shape)

        # Pass the result through the decoder
        # Break down decoder into its 7 steps
        d1 = self.decoder[0](x)
        print("d1: ", d1.shape)
        d2 = self.decoder[1](d1)
        print("d2: ", d2.shape)
        d3 = self.decoder[2](d2)
        print("d3: ", d3.shape)
        d4 = self.decoder[3](d3)
        print("d4: ", d4.shape)
        d5 = self.decoder[4](d4)
        print("d5: ", d5.shape)

        # d6 = self.decoder[5](d5)
        #x = self.decoder(d5)
        # print("de: ", d6.shape)
        


        # Pass the result through the final convolution
        x = self.final_conv(d5)
        print("Out: ", x.shape)

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
                #print("Output shape: ", outputs.shape)
                print("Target shape: ", targets.shape)
                print("Input shape: ", inputs.shape)
                optimizer.zero_grad()
                # Convert to real
                inputs = torch.view_as_real(inputs)
                print(inputs.shape)
                exit()
                outputs = self(inputs)
                # Convert back to complex
                outputs = torch.view_as_complex(outputs)

                # The output from the model is a mask that represents the difference between the input and the target
                # We can use the difference to get both left and right channels
                
                output_stft = inputs * outputs
                
                # Compare this to the target stft

                # We compare the istft to the target
                # XD --> output stft
                # targets --> stft
                loss = criterion(output_stft, targets)

                loss.backward()
                optimizer.step()

             # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                for i, (targets, inputs, angle) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = self(inputs)
                    outputs = inputs * outputs
                    val_loss += criterion(outputs, targets).item()
                val_loss /= len(val_loader)  # Calculate average validation loss

            scheduler.step()
            if optimizer.param_groups[0]['lr'] < 0.0001:
                optimizer.param_groups[0]['lr'] = 0.0001
            print(f'Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}')
            