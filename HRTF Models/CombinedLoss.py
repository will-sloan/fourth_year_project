import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom loss function that combines time domain and frequency domain losses.
        
        Args:
        - alpha (float): Weight factor for the time domain loss and frequency domain loss.
                         `alpha` balances the contribution of each domain's loss to the total loss.
                         0 <= alpha <= 1. When alpha = 0.5, the losses are equally weighted.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.time_loss = nn.MSELoss()

    def forward(self, predicted, target):
        # Time-domain loss
        loss_time = self.time_loss(predicted, target)
        
        # Transform to frequency domain
        predicted_fft = torch.fft.fft(predicted, dim=-1)
        target_fft = torch.fft.fft(target, dim=-1)
        
        # Magnitude loss
        predicted_mag = torch.abs(predicted_fft)
        target_mag = torch.abs(target_fft)
        loss_mag = F.mse_loss(predicted_mag, target_mag)
        
        # Phase loss
        predicted_phase = torch.angle(predicted_fft)
        target_phase = torch.angle(target_fft)
        # Phase difference can be wrapped, ensure minimum distance in phase considering wrapping
        phase_diff = torch.abs(torch.atan2(torch.sin(predicted_phase - target_phase), torch.cos(predicted_phase - target_phase)))
        loss_phase = torch.mean(phase_diff ** 2)  # Mean squared error of phase difference
        
        # Combine losses, weighted by alpha
        combined_loss = (1 - self.alpha) * loss_time + self.alpha * (loss_mag + loss_phase)
        return combined_loss
