import os 
from torch.utils.data import Dataset
import torch
import utils

class StockDataset(Dataset):
    def __init__(self, ticker):
        self.dataset = utils.get_dataset(ticker) 

    def __len__(self):
        print(self.dataset)
        return len(self.dataset.index)