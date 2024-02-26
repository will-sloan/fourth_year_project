from HRIRDataset import HRIRDataset
from torch.utils.data import DataLoader
import torch

def create_dataset():
    sofa_file = '/workspace/fourth_year_project/HRTF Models/sofa_hrtfs/RIEC_hrir_subject_001.sofa'
    hrir_dataset = HRIRDataset(baseline_angles=[0])
    for i in range(1,20):
        hrir_dataset.load(sofa_file.replace('001', str(i).zfill(3)))
    return hrir_dataset

def get_data_loaders(hrir_dataset):
    train_size = int(0.7 * len(hrir_dataset))
    val_size = int(0.2 * len(hrir_dataset))
    test_size = len(hrir_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(hrir_dataset, [train_size, val_size, test_size])

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    return train_loader, val_loader, test_loader

def normalize(tensor):
    """
    Normalize a tensor to have values between -1 and 1
    """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 2 - 1

def add_angle(tensor, angle):
    """
    Add angle to the channel dimension of the tensor
    [batch_size, 1, seq_len] -> [batch_size, 2, seq_len]

    Tensor is shape [32, 1, 512]
    Angle is shape [32]
    Expand angle to match the sequence length [32, 1, 512]
    Concatenate along the channel dimension [32, 2, 512]
    """
    angle_expanded = angle.unsqueeze(-1).unsqueeze(1).expand(-1, 1, tensor.size(2))
    
    # Concatenate the expanded angle tensor with the original tensor along the channel dimension
    tensor_with_angle = torch.cat([tensor, angle_expanded], dim=1)
    
    return tensor_with_angle

def add_sos_and_eos(batch_sequences, sos_value, eos_value):
    """
    Extend each sequence in the batch to include distinct SOS and EOS tokens.
    
    Args:
        batch_sequences (Tensor): Input tensor of shape [batch_size, 2, seq_len].
        sos_value (int or float): The value to use for the SOS token.
        eos_value (int or float): The value to use for the EOS token.
        
    Returns:
        Tensor: Modified batch with SOS at the beginning and EOS at the end, shape [batch_size, 2, seq_len + 2].
    """
    batch_size, channels, seq_len = batch_sequences.shape
    # Create tensors for SOS and EOS tokens for the whole batch
    sos_tokens = torch.full((batch_size, channels, 1), sos_value, dtype=batch_sequences.dtype, device=batch_sequences.device)
    eos_tokens = torch.full((batch_size, channels, 1), eos_value, dtype=batch_sequences.dtype, device=batch_sequences.device)
    
    # Concatenate the SOS token at the beginning and the EOS token at the end
    extended_sequences = torch.cat([sos_tokens, batch_sequences, eos_tokens], dim=-1)
    
    return extended_sequences



def add_sos_and_eos_in_place(batch_sequences, sos_value, eos_value):
    """
    Replace the first and last values in each sequence of the batch with SOS and EOS tokens, respectively,
    keeping the sequence length unchanged.
    
    Args:
        batch_sequences (Tensor): Input tensor of shape [batch_size, 2, seq_len].
        sos_value (int or float): The value to use for the SOS token.
        eos_value (int or float): The value to use for the EOS token.
        
    Returns:
        Tensor: Modified batch with the first and last values replaced by SOS and EOS, respectively,
                shape [batch_size, 2, seq_len].
    """
    # Directly replace the first value of each sequence with the SOS token
    batch_sequences[:, :, 0] = sos_value
    
    # Directly replace the last value of each sequence with the EOS token
    batch_sequences[:, :, -1] = eos_value
    
    return batch_sequences

def remove_data(tensor, start_index, end_index):
    """
    Remove data from the tensor along the sequence dimension.
    ex: [batch_size, 2, seq_len] -> [batch_size, 2, start_index --> End_index]
    """
    return tensor[:, :, start_index:end_index]

def preprocess(src, target, angle, device):
    """
    Preprocess the src, target, and angle tensors for training.

    Keeps left channel, adds a channel dimension, converts to float, and normalizes.
    Then adds angle to channel dimension. 
    """
    # Move to the specified device
    src, target, angle = src.to(device), target.to(device), angle.to(device)
    # Select the left channel and add a channel dimension
    # src = src[:, 0, :].unsqueeze(1)
    target = target[:, 0, :].unsqueeze(1)
    # Convert to float and normalize
    src = normalize(src.float())
    target = normalize(target.float())
    src = src.float()
    target = target.float()
    angle = angle.float()

    # src = add_angle(src, angle)
    # target_to_pass = add_angle(target, angle)

    src = remove_data(src, 64, 256)
    target = remove_data(target, 64, 256)
    target_to_pass = target
    # print(src.shape, target.shape, angle.shape)
    # target = add_angle(target, angle)

    # src = add_sos_and_eos_in_place(src, sos_value=-2, eos_value=-3)
    # target = add_sos_and_eos_in_place(target, sos_value=-2, eos_value=-3)

    # src = src.permute(0, 2, 1)
    # target = target.permute(0, 2, 1)
    # print(src.shape, target.shape, angle.shape) # torch.Size([32, 3, 192]) torch.Size([32, 1, 192]) torch.Size([32])

    return src, angle, target, target_to_pass