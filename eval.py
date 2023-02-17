import argparse
import torch
import utils
from torch.utils.data import DataLoader
import model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot(args):
    test_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.start, end=args.end, train_set=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    stock_net = model.StockNet(in_features=args.in_features, out_features=args.out_features).to(device)

    stock_net.load_state_dict(torch.load(args.model))

    predictions = []
    labels = []
    total_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            features, y = batch
            y = y.float().to(device)

            input = torch.stack((features.open, features.high, features.low, features.close,  features.high_low_diff, features.close_open_diff, features.std, features.MAs.MA_3d, features.MAs.MA_7d, features.MAs.MA_14d, features.MAs.MA_21d)).float().to(device)
            input = torch.transpose(input, 0, 1)
            output = stock_net(input).flatten()


            for prediction in output:
                predictions.append(prediction.item())
            for truth in y:
                labels.append(truth.to('cpu'))

    utils.save_plot(args.start, args.ticker, predictions, labels)
    rmse, mape = utils.eval(predictions, labels)
    print(rmse, mape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./model_params/NKE.pth')
    parser.add_argument('--ticker', '-t', type=str, default='NKE')
    parser.add_argument('--start', type=str, default='2022-01-01')
    parser.add_argument('--end', type=str, default='2023-02-17')
    parser.add_argument('--interval', '-i', type=str, help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='1d')
    parser.add_argument('--MA_intervals', '-m', nargs='+',help='list of intervals of moving average(unit: day)', default=[3, 7, 14, 21])
    parser.add_argument('--std_interval', '-s', help='interval of std(unit: day)', type=int, default=7)
    parser.add_argument('--download','-d', action='store_true')
    parser.add_argument('--drop_head', help='drop the first drop_head rows of the csv file, since there are no value of MA and std for the first couple of days', type=int, default=21)

    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--in_features', type=int, default=11)
    parser.add_argument('--out_features', type=int, default=1)

    args = parser.parse_args()

    plot(args)