import numpy as np
import torch
from torch.utils.data import Dataset

import random
random.seed(42)
np.random.seed(42)

class StockData(Dataset):
    def __init__(self,train=True):
        if train:
            self.X = np.load('data/processed/trainX.npy',allow_pickle=True)
            self.Y = np.load('data/processed/trainY.npy',allow_pickle=True)

        else:
            self.X = np.load('data/processed/testX.npy',allow_pickle=True)
            self.Y = np.load('data/processed/testY.npy',allow_pickle=True)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self,idx):

        x = self.X[idx]
        y = self.Y[idx]

        assert not any(x[x!=x]) ,'NaN Here'

        x = torch.from_numpy(x).float()
        y = torch.Tensor([y])
        return x,y