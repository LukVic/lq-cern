from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, Tensor
import numpy as np

class lq_DataSet(Dataset):
    """
    Class implementing pytorch Dataset class, serves as an iterator over the dataset that is passed to it
    and also transforms the samples such that they are scaled properly for the algorithm to work optimally
    """
    def __init__(self, X : np.array, y : np.array, weights: np.array):
        self.X = from_numpy(X).float()
        self.y = from_numpy(y).long()
        self.weights = from_numpy(weights).float()
        # self.weights = self.weights + self.weights.min().abs()
        self.transform_data()

    def __len__(self):
        return self.X.shape[0]

    def transform_data(self):
        tss = TorchStandardScaler()
        tss.fit(self.X)
        # print(f"mean {tss.mean}, std {tss.std}")
        tss.transform(self.X)


    def __getitem__(self, idx : int):
        return self.X[idx,:], self.y[idx], self.weights[idx]


class TorchStandardScaler():
    "Scales the samples to be more uniform when training"

    def fit(self, x : Tensor):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
    def transform(self, x : Tensor):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x
