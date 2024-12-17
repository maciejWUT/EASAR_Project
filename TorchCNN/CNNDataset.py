import numpy as np
import torch
from torch.utils.data import Dataset

class CNNDataset(Dataset):
    def __init__(self, one_hots, dirs, transform=None):
        self.one_hots = one_hots
        self.dirs = dirs
        self.transform = transform

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        img_path = self.dirs[idx]
        image = np.load(img_path)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        one_hot = torch.tensor(self.one_hots[idx])
        if self.transform:
            image = self.transform(image)
        return image, one_hot