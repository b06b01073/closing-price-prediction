import argparse
import utils
from dataset import StockDataset

def train(args):
    dataset = StockDataset(ticker=args.ticker)

    print(len(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', type=str, help='The ticker of company', default='NKE')
    parser.add_argument('--interval', '-i', type=str, help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='1d')
    parser.add_argument('--period', '-p', help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='5y')
    args = parser.parse_args()

    train(args)     