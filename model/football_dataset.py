import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)


class FootballDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data.drop(
            columns=["game_id",
                     "date",
                     "competition_id",
                     "home_club_name",
                     "away_club_name",
                     "competition_type"],
            inplace=True)
        self.data.dropna(inplace=True)
        # self.data.fillna(0, inplace=True)

        # transform all except result columns
        x_scaled = preprocessing.MinMaxScaler().fit_transform(self.data)

        # concat scaled and result columns
        self.data = pd.DataFrame(x_scaled)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = sample[:-3].values.astype(dtype=float)
        target = sample[-3:].values.astype(dtype=float)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
