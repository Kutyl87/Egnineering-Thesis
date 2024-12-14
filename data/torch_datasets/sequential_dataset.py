import torch
from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __getitem__(self, idx):
        if idx > self.sequence_length - 1:
            idx_start = idx - self.sequence_length + 1
            x = self.X[idx_start:(idx + 1):]
        else:
            padding = self.X[0].repeat(self.sequence_length - idx - 1, 1)
            x = self.X[0:(idx + 1):]
            x = torch.cat((padding, x), dim=0)
        return x, self.y[idx]

    def __len__(self):
        return len(self.X)
