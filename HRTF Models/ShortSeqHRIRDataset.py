# Only has the HRIR sequences between 50 and 300 timesteps

from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np
import torch

class ShortSeqHRIRDataset(Dataset):

    def __init__(self, ground_truths=[0]):
        self.baseline = []
        self.targets = []
        self.ground_truths = ground_truths

    def load(self, sofa_path):
        sofa = SOFAFile(sofa_path, 'r')
        sourcePositions = sofa.getVariableValue('SourcePosition')
        data = sofa.getDataIR()

        # Get indexes of the source positions when the elevation is 0
        i = np.where(sourcePositions[:,1] == 0)[0]
        # get all the angles and hrirs for the 0 elevation
        angles = sourcePositions[i,0]
        hrirs = data[i,:,:]

        # Get indexes where elevatio is 0 and azimuth is 0
        ii = np.where(sourcePositions[:,0] == 0)[0]
        # Get overlap between i and ii, this is the baseline
        iii = np.intersect1d(i, ii)[0]
        # Get the baseline and add it to the list
        baseline = data[iii,:,:]
        baseline_short = torch.from_numpy(baseline[:,50:300])
        
        
        self.baseline.append(baseline_short)
        for index in range(len(i)):
            # Add a tuple (index_of_baseline, hrir, target_angle)
            hrir_short = torch.from_numpy(hrirs[index,:,50:300])
            self.targets.append((len(self.baseline) - 1, hrir_short, torch.tensor(angles[index])))

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        baseline_index, hrir, angle = self.targets[index]
        return self.baseline[baseline_index], hrir, angle