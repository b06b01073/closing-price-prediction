import yfinance as yf
import pandas as pd
import os
from dataset import StockDataset

def get_dataset(ticker, interval, MA_intervals, std_interval, download, drop_head, start, end, train_set):
    dataset_type = 'train' if train_set else 'test'
    path = f'data/{ticker}_{dataset_type}.csv'
    path = get_path(path)

    if os.path.exists(path) and not download:
        dataset = pd.read_csv(path)
    else:
        dataset = yf.download(tickers=ticker, start=start, end=end, interval=interval)
        insert_MAs(dataset, MA_intervals)
        insert_std(dataset, std_interval)
        dataset.to_csv(path)    

    dataset = dataset.iloc[drop_head-1:] 

    dataset = StockDataset(dataset, MA_intervals, std_interval, train_set)

    return dataset

def insert_MAs(dataset, MA_intervals):
    for interval in MA_intervals:
        window = int(interval) # drop the time unit
        dataset[f'MA_{interval}d'] = dataset['Close'].rolling(window=window).mean()


def insert_std(dataset, std_interval):
    dataset[f'std_{std_interval}d'] = dataset['Close'].rolling(window=std_interval).std()


def get_path(path):
    dir_path = os.path.dirname(__file__) 
    path = os.path.join(dir_path, path)
    return path