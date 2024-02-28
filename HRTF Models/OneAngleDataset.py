from torch.utils.data import Dataset
from pysofaconventions import SOFAFile
import numpy as np
import torch

class BasicDataset(Dataset):

    def __init__(self, baseline_angles=[0,90]):
        # self.baseline = []
        self.targets = []
        self.baseline_angles = baseline_angles

    def load(self, sofa_path):
        sofa = SOFAFile(sofa_path, 'r')
        sourcePositions = sofa.getVariableValue('SourcePosition')
        data = sofa.getDataIR()

        # Get indexes of the source positions when the elevation is 0
        elev0 = np.where(sourcePositions[:,1] == 0)[0]
        # get all the angles and hrirs for the 0 elevation

        angle0 = np.where(sourcePositions[:,0] == 0)[0]
        angle90 = np.where(sourcePositions[:,0] == 90)[0]
        angle45 = np.where(sourcePositions[:,0] == 45)[0]

        indexes0 = np.intersect1d(elev0, angle0)
        indexes90 = np.intersect1d(elev0, angle90)
        indexes45 = np.intersect1d(elev0, angle45)

        hrirs0 = data[indexes0,:,:]
        hrirs90 = data[indexes90,:,:]
        hrirs45 = data[indexes45,:,:]

        # Length should all be the same and equal 1
        assert len(hrirs0) == len(hrirs90) == len(hrirs45) == 1

        # convert angles to tensors
        hrirs0 = torch.tensor(hrirs0, dtype=torch.float32)
        hrirs90 = torch.tensor(hrirs90, dtype=torch.float32)
        hrirs45 = torch.tensor(hrirs45, dtype=torch.float32)
        angle0 = torch.tensor(0., dtype=torch.float32)
        angle90 = torch.tensor(9., dtype=torch.float32)
        angle45 = torch.tensor(45., dtype=torch.float32)
        # Convert from (1, 2, 512) to (2, 512)
        hrirs0 = hrirs0.squeeze(0)
        hrirs90 = hrirs90.squeeze(0)
        hrirs45 = hrirs45.squeeze(0)

        output = (hrirs0, angle0, hrirs90, angle90, hrirs45, angle45)
        # print(hrirs0.shape)
        self.targets.append(output)

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.targets[index]