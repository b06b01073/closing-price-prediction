import yfinance as yf
import pandas as pd
import os

def get_dataset(ticker):
    path = get_path(f'./data/{ticker}.csv')
    data = pd.read_csv(path) if os.path.exists(path) else yf.download(tickers=ticker, period='5y', interval='1d')
    data.to_csv(path)


def get_path(path):
    dir_path = os.path.dirname(__file__) 
    path = os.path.join(dir_path, path)
    return path