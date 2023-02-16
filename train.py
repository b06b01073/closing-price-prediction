import argparse
import utils
from dataset import StockDataset

def main(args):
    dataset = StockDataset(ticker=args.ticker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, help='The ticker of company', default='NKE')
    args = parser.parse_args()

    main(args)