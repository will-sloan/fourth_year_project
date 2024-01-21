import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



class SequenceModel(nn.Module):
    def __init__(self, input_size=(32001) + 1, hidden_size=10000, output_size=1):
        super(SequenceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size + input_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size + input_size, hidden_size)  # added hidden layer
        self.relu3 = nn.LeakyReLU()  # added LeakyReLU activation
        self.fc4 = nn.Linear(hidden_size + input_size, hidden_size)  # added hidden layer
        self.relu4 = nn.LeakyReLU()  # added LeakyReLU activation
        self.fc5 = nn.Linear(hidden_size + input_size, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = self.relu1(out1)
        out2 = self.fc2(torch.cat((x, out1), dim=1))
        out2 = self.relu2(out2)
        out3 = self.fc3(torch.cat((x, out2), dim=1))  # added forward pass for new hidden layer
        out3 = self.relu3(out3)  # added LeakyReLU activation
        out4 = self.fc4(torch.cat((x, out3), dim=1))  # added forward pass for new hidden layer
        out4 = self.relu4(out4)  # added LeakyReLU activation
        out5 = self.fc5(torch.cat((x, out4), dim=1))  # final layer now receives input from last hidden layer
        return out5

    def train_loop(self, sp, batch_size, epochs, writer=None, loop_num=0, name='time_model_checkpoints_right'):

        
        scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        criterion = nn.MSELoss()
        sp.load_chunk()
        print("Loaded chunk")
        #sp.normalize()
        # # Divide the dataset into training, validation, and testing
        train_size = int(0.8 * len(sp))
        val_size = int(0.1 * len(sp))
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
            # self.optimizer.zero_grad()  # Initialize gradients once before the loop
            print("Train: ")
            for i, (input, angle, value, index) in enumerate(dataloader):
                self.optimizer.zero_grad() # 

                print(input.shape, angle.shape, value.shape, index.shape)
                #inputs = input.cuda()
                outputs = self(torch.cat((input, angle), dim=1).cuda())
                loss = criterion(outputs, value)
                loss.backward()
                
                self.optimizer.step()

                print(f"\tbatch {i}: {loss.item()}")
                if writer is not None:
                    writer.add_scalar('training loss', loss, epoch * len(dataloader) + i)
                if i % 3 == 0: 
                   torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        # Include any other states you want to save
                    }, f'/workspace/extension/unet/{name}/model_loopnum_{loop_num}_{epoch}_batch_{i}.pt')
                #exit()
            print("Done train")
            # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                print("Val: ")
                for i, (input, angle, value, index) in enumerate(val_loader):
                    self.optimizer.zero_grad() # 
                    outputs = self(torch.cat((input, angle), dim=1).cuda())
                    loss = criterion(outputs, value)
                    loss.backward()
                    
                    self.optimizer.step()
                     
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
            model_name = f'/workspace/extension/unet/{name}/model_loopnum_{loop_num}_{epoch}_afterval.pt'
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
            for i, (input, angle, value, index) in enumerate(test_loader):
                self.optimizer.zero_grad() # 
                outputs = self(torch.cat((input, angle), dim=1).cuda())
                loss = criterion(outputs, value)
                loss.backward()
                
                self.optimizer.step()
                print(f"\ttest-batch {i}: {loss.item()}")
                test_loss += loss.item()
        #print(f'Test Loss: {test_loss:.6f}')
        #print(f"Test length: {len(test_loader)}")
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.6f}')

        # self.train()  # Set the model back to training mode
        return model_name
            