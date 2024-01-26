import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class AutoEncoder(nn.Module):
    def __init__(self, input_channels=2, out_channels=2, lr=0.01):
        super(AutoEncoder, self).__init__()
        def conv_block(in_channels, out_channels):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Tanh(),  # Replaced ReLU with Tanh
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)  # Replaced ReLU with LeakyReLU
            ]
            
            return nn.Sequential(*layers)  

        self.encoder = nn.Sequential(
            conv_block(input_channels, 32),
            nn.MaxPool2d(2), 
            conv_block(32, 64),
            nn.MaxPool2d(2), 
            conv_block(64, 128),
        )

        self.bottleneck = conv_block(128 + 1, 256) 

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), 
            conv_block(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            conv_block(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1), 
            conv_block(32, 32),
        )

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01, betas=(0.95, 0.999))
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        

    def forward(self, x, angle):
        # Pass input through the encoder
        x = self.encoder(x)
        angle_tensor = angle.float() / 180.0
        angle_tensor = angle_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        angle_tensor = angle_tensor.expand(-1, -1, x.shape[2], x.shape[3])
        angle_tensor = angle_tensor.cuda()
        x = torch.cat([x, angle_tensor], dim=1)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)

        # Return to original shape
        x = x.permute(0, 2, 3, 1)

        return x
    


    def train_loop(self, sp, batch_size, epochs, writer=None, loop_num=0, name='right_model_checkpoints', bonus=None):
        #dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        #optimizer = optim.Adam(self.parameters(), lr=lr)
        # Add momentum to the optimizer
        
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        # Cross entropy
        # criterion = nn.CrossEntropyLoss()
        sp.load_chunk()
        print("Loaded chunk")
        #sp.normalize()
        # # Divide the dataset into training, validation, and testing
        train_size = int(0.7 * len(sp))
        val_size = int(0.2 * len(sp))
        test_size = len(sp) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(sp, [train_size, val_size, test_size])
        print(train_size, val_size, test_size)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
        model_name = ''
        import time

        # sleep for 10 seconds
        # print("Sleeping")
        # time.sleep(10)

        # exit()
        for epoch in range(1, epochs + 1):
            # load a fresh 1000 samples from each epoch
            

            # accumulation_steps = 4  # Change this to the number of steps you want to accumulate gradients over

            self.train()
            train_loss = 0.0
            # self.optimizer.zero_grad()  # Initialize gradients once before the loop
            #print("Train: ")
            for i, (mono, label, _, diffs, _) in enumerate(dataloader):
                self.optimizer.zero_grad() # 
                inputs, targets = mono.cuda(), diffs.cuda()
                outputs = self(inputs, label)
                outputs = outputs.contiguous()
                outputs = torch.view_as_complex(outputs)

                targets = targets.permute(0, 2, 3, 1)
                targets = targets.contiguous()
                # print(targets.shape)
                targets = torch.view_as_complex(targets)

                # Create a binary mask of the non-zero values in outputs or targets
                mask = (outputs != 0) | (targets != 0)

                # Apply the mask to the outputs and targets
                masked_outputs = outputs[mask]
                masked_targets = targets[mask]

                # scale both targets and outputs
                real_loss = criterion(masked_outputs.real, masked_targets.real)
                imag_loss = criterion(masked_outputs.imag, masked_targets.imag)
                loss = real_loss + imag_loss
                
                # loss = loss  # Normalize the loss because it's accumulated over multiple batches
                train_loss += loss.item()
                # loss.backward()  # Backpropagate the loss
                # Backward pass for real and imaginary losses separately
                real_loss.backward(retain_graph=True)  # Calculate the gradients for the real loss
                imag_loss.backward()  # Calculate the gradients for the imaginary loss

                # if (i+1) % accumulation_steps == 0:  # Only update the weights once every accumulation_steps batches
                #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
                
                self.optimizer.step()
                      # Zero the gradients after updating the weights

                # print(f"\tbatch {i}: {loss.item()}")
                if writer is not None:
                    writer.add_scalar('training loss', loss, epoch * len(dataloader) + i)
                # if i % 3 == 0: 
                #    torch.save({
                #         'epoch': epoch,
                #         'model_state_dict': self.state_dict(),
                #         'optimizer_state_dict': self.optimizer.state_dict(),
                #         # Include any other states you want to save
                #     }, f'/workspace/extension/unet/{name}/model_loopnum_{loop_num}_{epoch}_batch_{i}.pt')
                #exit()
            #print("Done train")
            # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                #print("Val: ")
                for i, (mono, label, _, diffs, _) in enumerate(val_loader):
                    self.optimizer.zero_grad() # 
                    inputs, targets = mono.cuda(), diffs.cuda()
                    outputs = self(inputs, label)
                    outputs = outputs.contiguous()
                    outputs = torch.view_as_complex(outputs)

                    targets = targets.permute(0, 2, 3, 1)
                    targets = targets.contiguous()
                    # print(targets.shape)
                    targets = torch.view_as_complex(targets)
                    # scale both targets and outputs
                    real_loss = criterion(outputs.real, targets.real)
                    imag_loss = criterion(outputs.imag, targets.imag)
                    loss = real_loss + imag_loss
                     
                    #print(f"\tval-batch {i}: {loss.item()}")
                    if writer is not None:
                        writer.add_scalar('validation loss', loss, epoch * len(dataloader) + i)
                    val_loss = val_loss + loss.item()
                val_loss /= len(val_loader)  # Calculate average validation loss
            #print("Done val")        
            scheduler.step()
            # if self.optimizer.param_groups[0]['lr'] < 0.0001:
            #     self.optimizer.param_groups[0]['lr'] = 0.0001
            train_loss /= len(dataloader)
            print(f'Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]}')

            # After every 2 epochs, save the model checkpoint
            if bonus is not None:
                model_name = f'/workspace/extension/unet/{name}/model_loopnum_{loop_num}_{epoch}_{bonus}.pt'
            else:
                model_name = f'/workspace/extension/unet/{name}/model_loopnum_{loop_num}_{epoch}_afterval.pt'
            if epoch % 25 == 0:
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
            #print("Test: ")
            for i, (mono, label, _, diffs, _) in enumerate(test_loader):
                self.optimizer.zero_grad() # 
                inputs, targets = mono.cuda(), diffs.cuda()
                outputs = self(inputs, label)
                outputs = outputs.contiguous()
                outputs = torch.view_as_complex(outputs)

                targets = targets.permute(0, 2, 3, 1)
                targets = targets.contiguous()
                # print(targets.shape)
                targets = torch.view_as_complex(targets)
                # scale both targets and outputs
                real_loss = criterion(outputs.real, targets.real)
                imag_loss = criterion(outputs.imag, targets.imag)
                loss = real_loss + imag_loss
                #print(f"\ttest-batch {i}: {loss.item()}")
                test_loss += loss.item()
        #print(f'Test Loss: {test_loss:.6f}')
        #print(f"Test length: {len(test_loader)}")
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.6f}')

        # self.train()  # Set the model back to training mode
        return model_name
            