from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np
import torch

class HRIRDataset(Dataset):

    def __init__(self, baseline_angles=[0,90,180,270]):
        # self.baseline = []
        self.targets = []
        self.baseline_angles = baseline_angles

    def load(self, sofa_path):
        sofa = SOFAFile(sofa_path, 'r')
        sourcePositions = sofa.getVariableValue('SourcePosition')
        data = sofa.getDataIR()

        # Get indexes of the source positions when the elevation is 0
        i = np.where(sourcePositions[:,1] == 0)[0]
        # get all the angles and hrirs for the 0 elevation
        angles = sourcePositions[i,0]
        hrirs = data[i,:,:]
        baselines = []

        for angle in self.baseline_angles:
            # Get indexes where elevatio is 0 and azimuth is angle
            ii = np.where(sourcePositions[:,0] == angle)[0]
            # Get overlap between i and ii, this is the baseline
            iii = np.intersect1d(i, ii)[0]
            # Get the baseline and add it to the list
            baseline = data[iii,:,:]
            baselines.append(torch.from_numpy(baseline))

        for index in range(len(i)):
            # Add a tuple (index_of_baseline, hrir, target_angle)
            self.targets.append((baselines, torch.from_numpy(hrirs[index,:,:]), torch.tensor(angles[index])))

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # baseline_index, hrir, angle = self.targets[index]
        # return self.baseline[baseline_index], hrir, angle
        # baselines, hrir, angle = self.targets[index]
        baselines, hrir, angle = self.targets[index]
        b0, b90, b180, b270 = baselines
        # Since baselines can be a list of tensors, we need to stack them
        # baselines = torch.cat(baselines, dim=0)
        # return baselines, hrir, angle
        return b0, b90, b180, b270, hrir, angle