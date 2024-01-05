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



class AudioCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(AudioCNN, self).__init__()
        # Encoder
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
        
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256*500, 1024)  # Bottleneck layer

        # Decoder
        self.fc2 = nn.Linear(1024, 256*500)  # Decoder fully connected layer
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.debn1 = nn.BatchNorm1d(128)
        self.derelu1 = nn.ReLU()
        self.dedropout1 = nn.Dropout(dropout_rate)
        
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.debn2 = nn.BatchNorm1d(64)
        self.derelu2 = nn.ReLU()
        self.dedropout2 = nn.Dropout(dropout_rate)
        
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.debn3 = nn.BatchNorm1d(32)
        self.derelu3 = nn.ReLU()
        self.dedropout3 = nn.Dropout(dropout_rate)
        
        self.deconv4 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.debn4 = nn.BatchNorm1d(16)
        self.derelu4 = nn.ReLU()
        self.dedropout4 = nn.Dropout(dropout_rate)
        
        self.deconv5 = nn.ConvTranspose1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.debn5 = nn.BatchNorm1d(8)
        self.derelu5 = nn.ReLU()
        self.dedropout5 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Encoder
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout5(self.relu5(self.bn5(self.conv5(x))))

        # Flatten the output from the encoder
        x = x.view(x.size(0), -1)

        # Bottleneck layer
        x = self.fc1(x)

        # Decoder
        x = self.fc2(x)

        # Reshape the output from the decoder fully connected layer
        x = x.view(x.size(0), 256, -1)

        x = self.dedropout1(self.derelu1(self.debn1(self.deconv1(x))))
        x = self.dedropout2(self.derelu2(self.debn2(self.deconv2(x))))
        x = self.dedropout3(self.derelu3(self.debn3(self.deconv3(x))))
        x = self.dedropout4(self.derelu4(self.debn4(self.deconv4(x))))
        x = self.dedropout5(self.derelu5(self.debn5(self.deconv5(x))))

        return x
    

    def train_loop(self, dataset, batch_size, epochs, lr):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        criterion = nn.MSELoss()
        #print("Starting training loop")

        for epoch in range(epochs):
            #print("Epoch", epoch)
            for i, (_, targets, _, inputs, angle, sr) in enumerate(dataloader):
                #inputs, targets = batch
                inputs, targets = inputs.cuda(), targets.cuda()
                #print("Loaded data")

                optimizer.zero_grad()
                outputs = self(inputs)
                #print("Ran model")
                #print(outputs.shape, targets.shape)
                #print(outputs[0,i,:10], targets[0,i,:10])
                loss = criterion(outputs, targets)
                #print("Calculated loss")
                loss.backward()
                optimizer.step()
            #print("Epoch", epoch, "loss", loss.item())
            scheduler.step()

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')
            
            # Save the model every 10 epochs
            if epoch % 10 == 0 and epoch != 0:
                print("Model saved at epoch", epoch)
                # Take first 4 digits of loss
                loss_str = str(loss.item())
                loss_str = loss_str[:5]
                torch.save(self.state_dict(), f'complex_cnn_loss_{loss_str}_{epoch}.pt')



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
print("Model created")
# Runs our training function
myTransformer.train_loop(dataset=audio_codes_dataset, batch_size=4, epochs=600, lr=0.01)