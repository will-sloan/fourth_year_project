# Creates and runs a simple CNN on the audio data


# Make sure we can import on the path we are on
import sys
sys.path.append('/workspace/fourth_year_project/MusicGen')

from MyAudioDataset import MyAudioDataset
from AudioCodesDataset import AudioCodesDataset
from audiocraft.models import CompressionModel
from audiocraft.models.encodec import InterleaveStereoCompressionModel
#from audiocraft.models import InterleaveStereoCompressionModel

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



class AudioCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # Increased filters
        self.bn5 = nn.BatchNorm1d(256)  # Increased filters
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256*500, 1024)  # Added fully connected layer
        self.fc2 = nn.Linear(1024, 8*500)  # Added fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        #print(x.size())

        x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(x.size())  # Print the size of the tensor here
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.size())
        x = x.view(-1, 8, 500)
        #print(x.size())
        
        return x

    def train_loop(self, dataset, batch_size, epochs, lr):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for i, (_, targets, _, inputs, angle, sr) in enumerate(dataloader):
                #inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = self(inputs)
                #print(outputs.shape, targets.shape)
                #print(outputs[0,i,:10], targets[0,i,:10])
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')
            
            # Save the model every 10 epochs
            if epoch % 10 == 0 and epoch != 0:
                print(f'Saving model at epoch {epoch}')
                # Take first 4 digits of loss
                loss_str = str(loss.item())
                loss_str = loss_str[:5]
                torch.save(self.state_dict(), f'simple_cnn_epoch_{loss_str}_{epoch}.pt')

if __name__ == "__main__":

    # Compression model, shortens 10secs to 8,500
    model = CompressionModel.get_pretrained('facebook/encodec_32khz')
    comp_model = InterleaveStereoCompressionModel(model).cuda()
    print("Compression model loaded")

    #mydataset = MyAudioDataset('/workspace/small_model_data3', 'recording_01_')
    audio_codes_dataset = AudioCodesDataset(comp_model)
    audio_codes_dataset.load_data('/workspace/90_degree_compress_tensors_10sec.pkl')
    #audio_codes_dataset.set_audio_dataset(mydataset)

    print("Dataset loaded")

    assert len(audio_codes_dataset) == 1710, "Dataset is not the right size"


    # Create transformer
    myTransformer = AudioCNN(dropout_rate=0.1).cuda()
    myTransformer.train()
