# A more complex CNN


# Make sure we can import on the path we are on
import sys
sys.path.append('/workspace/fourth_year_project/MusicGen')

from MyAudioDataset import MyAudioDataset
from AudioCodesDataset import AudioCodesDataset
from audiocraft.models import CompressionModel
from audiocraft.models.encodec import InterleaveStereoCompressionModel

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F




class FullyConnectedNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(4000, 4000),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc7 = nn.Linear(4000, 4000)  # No activation, dropout or batch normalization on the last layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x
    

    def train_loop(self, train_dataset, val_dataset, batch_size, epochs, lr, name):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        criterion = nn.MSELoss()
        #print("Starting training loop")
        last_20_losses = []

        for epoch in range(epochs):
            #print("Epoch", epoch)
            self.train() 
            for i, (_, targets, _, inputs, angle, sr) in enumerate(dataloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            last_20_losses.append(loss.item())
            if len(last_20_losses) > 20:
                last_20_losses.pop(0)
            running_avg_loss = sum(last_20_losses) / len(last_20_losses)

             # Validation
            self.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # No need to track gradients in validation
                val_loss = 0
                for i, (_, targets, _, inputs, angle, sr) in enumerate(val_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    inputs = inputs.view(inputs.size(0), -1)
                    targets = targets.view(targets.size(0), -1)
                    outputs = self(inputs)
                    val_loss += criterion(outputs, targets).item()
                val_loss /= len(val_loader)  # Calculate average validation loss

            scheduler.step()
            if optimizer.param_groups[0]['lr'] < 0.0001:
                optimizer.param_groups[0]['lr'] = 0.0001
            print(f'Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss}, Running Avg: {running_avg_loss}, LR: {scheduler.get_last_lr()[0]}')
            
            # Save the model every 10 epoch
            if epoch % 25 == 0 and epoch != 0:
                print("Model saved at epoch", epoch)
                # Take first 4 digits of loss
                loss_str = str(loss.item())
                loss_str = loss_str[:5]
                torch.save(self.state_dict(), f'/workspace/extension/nn_models/{name}_encoder_cnn_loss_{loss_str}_{epoch}.pt')

def test_model(myTransformer, test_dataset):
    myTransformer.eval()
    test_loss = 0
    criterion = nn.MSELoss()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for i, (_, targets, _, inputs, angle, sr) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            outputs = myTransformer(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_dataset)
    return avg_test_loss


import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Train a model')

# Add an argument
parser.add_argument('--name', type=str, help='The name to prepend to the saved model')

# Parse the arguments
args = parser.parse_args()



# Compression model, shortens 10secs to 8,500
model = CompressionModel.get_pretrained('facebook/encodec_32khz')
comp_model = InterleaveStereoCompressionModel(model).cuda()
print("Compression model loaded")

#mydataset = MyAudioDataset('/workspace/small_model_data3', 'recording_01_')
audio_codes_dataset = AudioCodesDataset(comp_model)
audio_codes_dataset.load_data('/workspace/90_degree_compress_tensors_10sec_augmented.pkl')
#audio_codes_dataset.set_audio_dataset(mydataset)

print("Dataset loaded")

assert len(audio_codes_dataset) == 5130, f"Dataset is not the right size, got {len(audio_codes_dataset)}"

# Split dataset into train, validation and test
from torch.utils.data import random_split

# Define the proportions for the split
train_proportion = 0.8 
val_proportion = 0.1
test_proportion = 0.1 

# Calculate the number of samples for train, validation and test
train_size = int(train_proportion * len(audio_codes_dataset))
val_size = int(val_proportion * len(audio_codes_dataset))
test_size = len(audio_codes_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(audio_codes_dataset, [train_size, val_size, test_size])


# Create transformer
myTransformer = FullyConnectedNN().cuda()
# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

myTransformer.apply(weights_init)

# Load from checkpoint
#myTransformer.load_state_dict(torch.load('/workspace/extension/encoder_models/encoder_cnn_loss_0.791_75.pt'))

myTransformer.train()
print("Model created")



# Runs our training function
myTransformer.train_loop(train_dataset=train_dataset, val_dataset=val_dataset, batch_size=8, epochs=1000, lr=0.01, name=args.name)

# Test the model
test_model(myTransformer, test_dataset)
