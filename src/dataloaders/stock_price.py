import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#This dataset loads a single stock into training
class StockPriceDataset(Dataset):

    # dataframe is the df
    # target is the target column]
    # features are the feature columns that get used in the LSTM
    # sequence length is the days in advance that get fed into the LSTM
    # daily_features
    def __init__(self, dataframe, target, features_series, features_daily, sequence_length=10):
        self.features_series = features_series
        self.features_daily = features_daily
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features_series].values).float()
        self.X_daily = torch.tensor(dataframe[features_daily].values).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length

    def __getitem__(self, i): 
        i = i + self.sequence_length
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x1 = self.X[i_start:i, :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i), :]
            x = torch.cat((padding, x), 0)
        x2 = self.X_daily[i]

        return x1,x2, self.y[i]

def get_loader(owlcv_df, batch_size = 8, shuffle=False):
    all_datasets = []
    for _,df in owlcv_df.groupby(by = 'Symbol'):
        dataset = StockPriceDataset(
            df,
            target='Return',
            features_series=['Open','Volume','High','Close','Adjusted Close','Return'],
            features_daily = ['Open','Day_of_week'],
            sequence_length=10
        )
        all_datasets.append(dataset)
    full_set = torch.utils.data.ConcatDataset(all_datasets)
    dataloader = DataLoader(full_set, batch_size=batch_size, shuffle=shuffle)
    return dataloader
