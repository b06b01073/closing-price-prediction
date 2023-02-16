import yfinance as yf
import pandas as pd
import os

def get_dataset(ticker):
    path = get_path(f'./data/{ticker}.csv')

    if os.path.exists(path):
        dataset = pd.read_csv(path)
    else:
        dataset = yf.download(tickers=ticker, period='5y', interval='1d')
        dataset.to_csv(path, index=False)    

    return dataset



def get_path(path):
    dir_path = os.path.dirname(__file__) 
    path = os.path.join(dir_path, path)
    return path