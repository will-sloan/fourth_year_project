import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, weighted_positions=None, weight=5):
        """
        Custom loss function that combines time domain and frequency domain losses, with optional weighting on specific positions.
        
        Args:
        - alpha (float): Weight factor for balancing time domain and frequency domain losses.
        - weighted_positions (list or None): Positions within the sequence where the loss should be weighted more heavily.
        - weight (float): The weighting factor to apply to the loss at the specified positions.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.weight = weight
        self.weighted_positions = torch.tensor(weighted_positions) if weighted_positions is not None else None
        self.time_loss = nn.MSELoss(reduction='none')

    def forward(self, predicted, target):
        # Calculate time domain loss
        loss_time = self.time_loss(predicted, target)
        
        # If weighted_positions are provided, apply additional weight to those positions
        if self.weighted_positions is not None:
            weights = torch.ones_like(loss_time)
            weights[:, :, self.weighted_positions] *= self.weight
            loss_time = loss_time * weights
            loss_time = torch.mean(loss_time)  # Take mean after applying weights
        else:
            loss_time = torch.mean(loss_time)  # Mean of losses if no weighting is applied
        
        # Frequency domain losses
        predicted_fft = torch.fft.fft(predicted, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # Magnitude loss
        predicted_mag = torch.abs(predicted_fft)
        target_mag = torch.abs(target_fft)
        loss_mag = F.mse_loss(predicted_mag, target_mag)
        
        # Phase loss
        predicted_phase = torch.angle(predicted_fft)
        target_phase = torch.angle(target_fft)
        phase_diff = torch.abs(torch.atan2(torch.sin(predicted_phase - target_phase), torch.cos(predicted_phase - target_phase)))
        loss_phase = torch.mean(phase_diff ** 2)  # MSE for phase difference
        
        # Combine losses, weighted by alpha
        combined_loss = (1 - self.alpha) * loss_time + self.alpha * (loss_mag + loss_phase)
        return combined_loss
