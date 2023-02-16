import os 
from torch.utils.data import Dataset
import torch
import utils
from collections import namedtuple
import numpy as np

class StockDataset(Dataset):
    def __init__(self, dataset, MA_intervals, std_interval):
        self.dataset = dataset 
        self.MA_intervals = MA_intervals
        self.std_interval = std_interval

    def fetch_data(self, index):
        row = self.dataset.iloc[index]

        StockData = namedtuple('StockData', ['High', 'Low', 'Close', 'Open', 'MAs', 'std'])

        MAs_cols = [f'MA_{interval}d'for interval in self.MA_intervals]
        MAs = namedtuple('MAs', MAs_cols)
        moving_averages = MAs(*row[MAs_cols])

        stock_data = StockData(
            High = row['High'],
            Low = row['Low'],
            Close = row['Close'],
            Open = row['Open'],
            MAs = moving_averages,
            std = row[f'std_{self.std_interval}d']
        )

        return stock_data

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, index):

        # ignore the last data point, since there are no ground truth(the data from the day after that)
        if index == len(self) - 1:
            index = np.random.randint(low=0, high=len(self) - 1)
        
        return self.fetch_data(index), self.dataset.iloc[index + 1]['Close']

    