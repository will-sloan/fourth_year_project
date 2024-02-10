from MainModel import MainModel
from torch.utils.data import DataLoader
from HRIRDataset import HRIRDataset

from torch import optim, nn
import torch

sofa_file = 'sofa_hrtfs/RIEC_hrir_subject_001.sofa'
hrir_dataset = HRIRDataset()
hrir_dataset.load(sofa_file)
# Create the model
model = MainModel(d_model=3)
# Set the model to training mode
model.train()
num_epochs = 10

# Create the DataLoader
dataloader = DataLoader(hrir_dataset, batch_size=2, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else exit())
model = model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
loss_function = nn.MSELoss()

# Set the model to training mode
model.train()

# Loop over each epoch
for epoch in range(num_epochs):
    # Initialize the epoch loss
    epoch_loss = 0.0

    # Loop over each batch
    for i, batch in enumerate(dataloader):
        # Get the src and tgt sequences from the batch
        #print(batch)
        src, tgt, angle = batch

        # Move data to the same device as the model
        src = src.to(device)
        tgt = tgt.to(device)
        angle = angle.to(device)
        # convert to floats
        angle = angle.float()
        src = src.float()
        tgt = tgt.float()

        # Zero the gradients
        optimizer.zero_grad()

        src = (src - src.mean()) / src.std()
        tgt = (tgt - tgt.mean()) / tgt.std()

        # Forward pass through the model
        output = model(src, tgt, angle)
        
        # remove the last feature dimension from output
        # [batch_size, d_model, seq_length] --> [batch_size, d_model-1, seq_length]
        output = output[:, :-1, :]
        #print(output.shape, tgt.shape)
        # Compute the loss
        loss = loss_function(output, tgt)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Accumulate the batch loss
        epoch_loss += loss.item()

    # Print the average loss for this epoch
    print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader)}')
