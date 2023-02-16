import argparse
import utils
from torch.utils.data import DataLoader

def train(args):
    train_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.training_start, end=args.training_end, train_set=True)
    test_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.simulation_start, end=args.simulation_end, train_set=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', type=str, help='The ticker of company', default='NKE')
    parser.add_argument('--interval', '-i', type=str, help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='1d')
    parser.add_argument('--MA_intervals', '-m', nargs='+',help='list of intervals of moving average(unit: day)', default=[7, 14, 21])
    parser.add_argument('--std_interval', '-s', help='interval of std(unit: day)', type=int, default=7)
    parser.add_argument('--download','-d', action='store_true')
    parser.add_argument('--drop_head', help='drop the first drop_head rows of the csv file, since there are no value of MA and std for the first couple of days', type=int, default=21)
    parser.add_argument('--training_start', type=str, default='2011-01-01')
    parser.add_argument('--training_end', type=str, default='2020-12-31')
    parser.add_argument('--simulation_start', type=str, default='2021-01-01')
    parser.add_argument('--simulation_end', type=str, default='2023-02-14')

    args = parser.parse_args()
    args.MA_intervals = [int(interval) for interval in args.MA_intervals]

    train(args)     