import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, data_path):
        super().__init__()

        data = np.load(data_path)

        self.images = torch.from_numpy(data['images']) # [106, 100, 100, 3]
        self.poses = torch.from_numpy(data['poses']) # [106, 4, 4]
        self.focal = torch.tensor(data['focal'], dtype=torch.float32)

        self.len = self.images.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return {
            'image' : self.images[idx],
            'pose' : self.poses[idx],
            'focal' : self.focal
        }