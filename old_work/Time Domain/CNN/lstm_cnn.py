# Creates and runs a simple CNN on the audio data


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



class AudioLSTMCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AudioLSTMCNN, self).__init__()
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

        # Batch_first means that the first dimension is the batch size
        self.lstm = nn.LSTM(256, 512, batch_first=True)  # New LSTM layer
        self.lstm.flatten_parameters()

        self.fc1 = nn.Linear(512*500, 1024)  # Added fully connected layer
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
        x = x.permute(0, 2, 1)  # LSTM expects input of shape (batch_size, seq_length, num_features)
        #print(x.size())
        
        x, _ = self.lstm(x)  # Apply LSTM here
        #print(x.size())
        x = x.permute(0, 2, 1)  # Permute back to (batch_size, num_features, seq_length)
        #print(x.size())
        x = x.contiguous().view(x.size(0), -1)  # Flatten the tensor
        #print(x.size())

        #print(x.size())

        #x = x.view(x.size(0), -1)  # Flatten the tensor
        #print(x.size())  # Print the size of the tensor here
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.size())
        x = x.view(-1, 8, 500)
        #print(x.size())
        
        
        return x
    def train_loop(self, dataset, batch_size, epochs, lr, use_mse=True):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        if use_mse:
            criterion = nn.MSELoss()
        else:
            # Use cosine loss
            criterion = nn.CosineEmbeddingLoss()

        for epoch in range(epochs):
            for i, (_, targets, _, inputs, angle, sr) in enumerate(dataloader):
                #inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                
                outputs = self(inputs)
                #print(outputs.shape, targets.shape)
                #print(outputs[0,i,:10], targets[0,i,:10])
                if use_mse:
                    loss = criterion(outputs, targets)
                else:
                    output = output.view(-1, 500)
                    target_codes = target_codes.view(-1, 500)
                    y = torch.ones((output.size(0),)).cuda()
                    loss = criterion(output, target_codes, y)

                
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            if optimizer.param_groups[0]['lr'] < 0.00001:
                optimizer.param_groups[0]['lr'] = 0.00001

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')
            
            # Save the model every 10 epochs
            if epoch % 60 == 0 and epoch != 0:
                print("Model saved at epoch", epoch)
                # Take first 4 digits of loss
                loss_str = str(loss.item())
                loss_str = loss_str[:5]
                torch.save(self.state_dict(), f'lstm_cnn_loss_{loss_str}_{epoch}.pt')



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
myTransformer = AudioLSTMCNN(dropout_rate=0.1).cuda()
myTransformer.train()
print("Model created")
# Runs our training function
myTransformer.train_loop(dataset=audio_codes_dataset, batch_size=4, epochs=600, lr=0.01)