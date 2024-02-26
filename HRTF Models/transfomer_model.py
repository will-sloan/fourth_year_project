
from HRIRDataset import HRIRDataset
from HRIRTransformerModel import HRIRTransformerModel
import matplotlib.pyplot as plt
from CombinedLoss import CombinedLoss
from utils import *
import torch.optim as optim
import torch
from MyTransformer import MyTransformer
from PositionalEncoding import PositionalEncoding
import torch.nn as nn
import math

class HRIRTransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, angle_dim, dropout=0.1):
        super(HRIRTransformerModel, self).__init__()
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.angle_encoder = nn.Linear(1, angle_dim)  # Encodes angle to a high-dimensional space
        self.encoder = nn.Linear(192 + angle_dim, d_model)  # Adjusted for concatenated angle
        self.decoder = nn.Linear(d_model, 192)  # Output HRIR prediction
        self.d_model = d_model
        self.angle_dim = angle_dim
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.angle_encoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, angle, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        angle_encoded = self.angle_encoder(angle.view(-1, 1)).unsqueeze(1)  # Assuming angle is a batch of scalars
        src = torch.cat((src, angle_encoded.expand(-1, src.size(1), -1)), -1)  # Concatenate along the feature dimension
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)
        tgt = self.pos_encoder(tgt * math.sqrt(self.d_model))
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.decoder(output)
        return output



hrir_dataset = create_dataset()

device = torch.device('cuda')
model = HRIRTransformerModel(d_model=192, nhead=2, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, angle_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed
# weighted_positions = list(range(50, 250))
weighted_positions = None
loss_function = CombinedLoss(alpha=0.2, weighted_positions=weighted_positions, weight=50).to(device) # Lower the alpha value to give more weight to the time
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

train_loader, val_loader, test_loader = get_data_loaders(hrir_dataset)

def train_epoch(model, data_loader, loss_function, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for src, target, angle in data_loader:
        # Adjust preprocessing to include shifting the target for teacher forcing
        src, angle, target, target_to_pass = preprocess(src, target, angle, device)
        src = src.to(device)
        target = target.to(device)
        target_to_pass = target_to_pass.to(device)
        # src - [batch_size, 2, 512], with angle, SOS, EOS
        # taret - [batch_size, 1, 512], with angle, SOS, EOS
        # if random generated number is less than 0.5, use target_to_pass
        if torch.rand(1) < 1.1:
            target_to_pass = target_to_pass.to(device)
            output = model(src, angle, target_to_pass)
        else:
            output = model(src)
        loss = loss_function(output, target)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Update the learning rate
    scheduler.step()
    
    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, loss_function, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, target, angle in data_loader:
            src, angle, target, target_to_pass = preprocess(src, target, angle, device)
            src = src.to(device)
            target = target.to(device)
            if torch.rand(1) < 1.1:
                target_to_pass = target_to_pass.to(device)
                output = model(src, angle, target_to_pass)
            else:
                output = model(src)
            
            # Compute loss
            loss = loss_function(output, target)
            
            total_loss += loss.item()
    return total_loss / len(data_loader)


num_epochs = 200 # Number of epochs to train for
model = model.to(device)  # Move model to the specified device
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, loss_function, optimizer, scheduler, device)
    val_loss = validate_epoch(model, val_loader, loss_function, device)
    
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

