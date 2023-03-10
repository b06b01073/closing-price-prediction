import yfinance as yf
import pandas as pd
import os
from dataset import StockDataset
import matplotlib.pyplot as plt
import numpy as np

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
        insert_volume_ratio(dataset)
        dataset.to_csv(path)    

    dataset = dataset.iloc[drop_head-1:] 
    return StockDataset(dataset, MA_intervals, std_interval, train_set)

def insert_volume_ratio(dataset):
    dataset['mean_volume'] = dataset['Volume'].expanding().mean()
    dataset['volume_ratio'] = dataset['Volume'] / dataset['mean_volume']

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

def save_plot(start, ticker, predictions, labels):
    x = [i for i in range(len(predictions))]
    plt.plot(x, predictions, color='blue', label='pred')
    plt.plot(x, labels, color='red', label='ground truth')
    plt.legend(loc='best')
    plt.ylabel('price')
    plt.xlabel(f'trading day since {start}')
    plt.title(ticker)
    plt.savefig(f'./result/{ticker}.png')

def eval(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    rmse = np.sqrt(np.mean((labels - predictions) ** 2))
    mape = np.mean(np.divide(np.abs(labels - predictions), labels)) * 100 # in percentage
    return rmse, mape