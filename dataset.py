import os 
from torch.utils.data import Dataset
import torch
import utils
from collections import namedtuple
import numpy as np

class StockDataset(Dataset):
    def __init__(self, dataset, MA_intervals, std_interval, train_set=False):
        self.dataset = dataset 
        self.MA_intervals = MA_intervals
        self.std_interval = std_interval
        self.train_set = train_set

    def fetch_data(self, index):
        row = self.dataset.iloc[index]

        StockData = namedtuple('StockData', ['high_low_diff', 'close_open_diff', 'MAs', 'std', 'volume', 'open','high', 'low', 'close', 'next_open', 'volume_ratio'])

        MAs_cols = [f'MA_{interval}d'for interval in self.MA_intervals]
        MAs = namedtuple('MAs', MAs_cols)
        moving_averages = MAs(*row[MAs_cols])


        stock_data = StockData(
            high_low_diff = (row['High'] - row['Low']),
            close_open_diff = (row['Close'] - row['Open']),
            MAs = moving_averages,
            std = row[f'std_{self.std_interval}d'],
            volume=row['Volume'],
            open = row['Open'],
            high = row['High'], 
            low = row['Low'],
            close = row['Close'],
            next_open = self.dataset.iloc[index + 1]['Open'],
            volume_ratio=row['volume_ratio']
        )

        return stock_data

    def __len__(self):
        # dataset的最後一筆資料為前一天的資料
        return len(self.dataset.index) - 1

    def __getitem__(self, index):
        return self.fetch_data(index), self.dataset.iloc[index + 1]['Close']
        