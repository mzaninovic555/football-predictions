import pandas as pd
import torch
from torch.utils.data import Dataset


class FootballDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.dropna(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = sample[6:-3].values.astype(dtype=int)
        target = sample[-3:].values.astype(dtype=int)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
