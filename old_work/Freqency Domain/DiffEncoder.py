import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



class AutoEncoder(nn.Module):
    def __init__(self, input_channels=2, out_channels=2, dropout_rate=0.01, lr=0.01):
        super(AutoEncoder, self).__init__()
        def conv_block(in_channels, out_channels, use_dropout=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return nn.Sequential(*layers)  
        
        def upconv_block(in_channels, out_channels, use_dropout=False, last_layer=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                
            ]
            if last_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(0.2, inplace=True))
            return nn.Sequential(*layers)  

        self.encoder = nn.Sequential(
            conv_block(input_channels, 128, use_dropout=False),
            nn.MaxPool2d(2), 
            conv_block(128, 256, use_dropout=False),
            nn.MaxPool2d(2), 
            conv_block(256, 512, use_dropout=False),
        )

        #self.angle_to_index = {0.0: 0, 15.0: 1, 30.0: 2, 45.0: 3, 60.0: 4, 75.0: 5, 90.0: 7, 105.0: 8, 120.0: 9, 135.0: 10, 150.0: 11, 165.0: 12, 180.0: 13}
        # Add one channel for the angle
        # self.bottleneck = conv_block(512 + 1, 768) # 256 + 512 = 768

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512 + 1, 512, kernel_size=2, stride=2), 
            upconv_block(512, 512, use_dropout=False, last_layer=False),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), 
            conv_block(256, 256, use_dropout=False, last_layer=False),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1), 
            conv_block(128, 64, use_dropout=False, last_layer=True),
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.3)
        

    def forward(self, x, angle):
        # Pass input through the encoder
        x = self.encoder(x)
        angle_tensor = angle.float() / 180.0
        angle_tensor = angle_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        angle_tensor = angle_tensor.expand(-1, -1, x.shape[2], x.shape[3])
        angle_tensor = angle_tensor.cuda()
        x = torch.cat([x, angle_tensor], dim=1)
        # x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)

        # Return to original shape
        x = x.permute(0, 2, 3, 1)

        return x
    

    def predict(self, mono_freq, angle):
        self.eval()
        mono_freq = mono_freq.cuda()
        mono_freq = torch.view_as_real(mono_freq)
        mono_freq = mono_freq.permute(0, 3, 1, 2)
        angle_tensor = angle.float() / 180.0
        angle_tensor = angle_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        angle_tensor = angle_tensor.expand(-1, -1, mono_freq.shape[2], mono_freq.shape[3])
        angle_tensor = angle_tensor.cuda()
        mono_freq = torch.cat([mono_freq, angle_tensor], dim=1)
        outputs = self(mono_freq, angle)
        outputs = outputs.contiguous()
        outputs = torch.view_as_complex(outputs)
        outputs = outputs.permute(0, 2, 3, 1)
        return outputs


    def train_loop(self, sp, batch_size, epochs, writer=None, loop_num=0, name='left'):
        #dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        #optimizer = optim.Adam(self.parameters(), lr=lr)
        # Add momentum to the optimizer
        
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        # criterion = nn.SmoothL1Loss()
        # criterion = nn.CosineEmbeddingLoss()
        #print("Starting training loop")
        max_norm = 1.0
        sp.load_chunk()
        print("Loaded chunk")
        #sp.normalize()
        # # Divide the dataset into training, validation, and testing
        train_size = int(0.6 * len(sp))
        val_size = int(0.3 * len(sp))
        test_size = len(sp) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])
        print(train_size, val_size, test_size)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        model_name = ''
        for epoch in range(1, epochs + 1):
            # load a fresh 1000 samples from each epoch
            

            # accumulation_steps = 4  # Change this to the number of steps you want to accumulate gradients over

            self.train()
            train_loss = 0.0
            self.optimizer.zero_grad()  # Initialize gradients once before the loop
            print("Train: ")
            for i, (mono, label, mixed, diffs) in enumerate(dataloader):
                mono, mixed, diffs = mono.cuda(), mixed.cuda(), diffs.cuda()

                mono = torch.view_as_real(mono)
                mono = mono.permute(0, 3, 1, 2)


                mask = self(mono, label)


                inputs = inputs.permute(0, 2, 3, 1)
                outputs = outputs.contiguous()
                inputs = inputs.contiguous()
                outputs = torch.view_as_complex(outputs)
                inputs = torch.view_as_complex(inputs)

                loss = criterion(mask, diffs)
                train_loss += loss.item()
                loss = loss / accumulation_steps  # Normalize the loss because it's accumulated over multiple batches
                
                loss.backward()

                if (i+1) % accumulation_steps == 0:  # Only update the weights once every accumulation_steps batches
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()  # Zero the gradients after updating the weights

                print(f"\tbatch {i}: {loss.item()}")
                if writer is not None:
                    writer.add_scalar('training loss', loss, epoch * len(dataloader) + i)
                if i % 3 == 0: 
                   torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        # Include any other states you want to save
                    }, f'/workspace/extension/unet/model_checkpoints4/model_{name}_loopnum_{loop_num}_{epoch}_batch_{i}.pt')
                #exit()
            print("Done train")
            # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                print("Val: ")
                for i, (mono, label, target) in enumerate(val_loader):
                    inputs, targets = mono.cuda(), target.cuda()
                    inputs = torch.view_as_real(inputs)
                    #self.zero_grad()
                    inputs = inputs.permute(0, 3, 1, 2)
                    outputs = self(inputs, label)
                    inputs = inputs.permute(0, 2, 3, 1)
                    outputs = outputs.contiguous()
                    inputs = inputs.contiguous()
                    outputs = torch.view_as_complex(outputs)
                    inputs = torch.view_as_complex(inputs)
                    output_stft = inputs * outputs
                    real_loss = criterion(output_stft.real, targets.real)
                    imag_loss = criterion(output_stft.imag, targets.imag)
                    loss = real_loss + imag_loss
                    print(f"\tval-batch {i}: {loss.item()}")
                    if writer is not None:
                        writer.add_scalar('validation loss', loss, epoch * len(dataloader) + i)
                    val_loss = val_loss + loss.item()
                val_loss /= len(val_loader)  # Calculate average validation loss
            print("Done val")        
            scheduler.step()
            # if optimizer.param_groups[0]['lr'] < 0.0001:
            #     optimizer.param_groups[0]['lr'] = 0.0001
            train_loss /= len(dataloader)
            print(f'Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}')

            # After every 2 epochs, save the model checkpoint
            model_name = f'/workspace/extension/unet/model_checkpoints4/model_{name}_loopnum_{loop_num}_{epoch}_afterval.pt'
            # if epoch % 10 == 0:
                # print("Saved model")
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # Include any other states you want to save
                }, model_name)

        #print("start test")
        # # # Test the model
        self.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            print("Test: ")
            for i, (mono, label, target) in enumerate(test_loader):
                inputs, targets = mono.cuda(), target.cuda()
                inputs = torch.view_as_real(inputs)
                #self.zero_grad()
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = self(inputs, label)
                inputs = inputs.permute(0, 2, 3, 1)
                outputs = outputs.contiguous()
                inputs = inputs.contiguous()
                outputs = torch.view_as_complex(outputs)
                inputs = torch.view_as_complex(inputs)
                output_stft = inputs * outputs
                real_loss = criterion(output_stft.real, targets.real)
                imag_loss = criterion(output_stft.imag, targets.imag)
                loss = real_loss + imag_loss
                print(f"\ttest-batch {i}: {loss.item()}")
                test_loss += loss.item()
        #print(f'Test Loss: {test_loss:.6f}')
        #print(f"Test length: {len(test_loader)}")
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.6f}')

        # self.train()  # Set the model back to training mode
        return model_name
            