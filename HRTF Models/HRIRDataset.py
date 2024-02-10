from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np
import torch

class HRIRDataset(Dataset):

    def __init__(self):
        self.baseline = []
        self.targets = []

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
        self.baseline.append(torch.from_numpy(baseline))
        for index in range(len(i)):
            # Add a tuple (index_of_baseline, hrir, target_angle)
            self.targets.append((len(self.baseline) - 1, torch.from_numpy(hrirs[index,:,:]), torch.tensor(angles[index])))

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        baseline_index, hrir, angle = self.targets[index]
        return self.baseline[baseline_index], hrir, angle