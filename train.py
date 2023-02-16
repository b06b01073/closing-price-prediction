import argparse
import utils
from torch.utils.data import DataLoader
import torch
import model
from torch import optim
from torch import nn 

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    train_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.training_start, end=args.training_end, train_set=True)
    test_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.test_start, end=args.test_end, train_set=False)

    print(f'len of train_set: {len(train_set)}')
    print(f'len of test_set: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)


    stock_net = model.StockNet(in_features=args.in_features, out_features=args.out_features).to(device)
    optimizer = optim.RMSprop(stock_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train(args, train_loader, stock_net, optimizer, criterion)
        test_loss = test(args, test_loader, stock_net, criterion)
        error_log(train_loss, test_loss, epoch)
        

def error_log(train_loss, test_loss, epoch):
    print(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}')

def train(args, train_loader, stock_net, optimizer, criterion):
    total_loss = 0
    for batch in train_loader:
        features, y = batch
        y = y.float().to(device)

        high_low_diffs = features.high_low_diff
        close_open_diffs = features.close_open_diff
        MA_7d = features.MAs.MA_7d
        MA_14d = features.MAs.MA_14d
        MA_21d = features.MAs.MA_21d
        std = features.std
        close = features.close

        input = torch.stack((high_low_diffs, close_open_diffs, std, MA_7d, MA_14d, MA_21d)).float().to(device)
        input = torch.transpose(input, 0, 1)

        output = stock_net(input).squeeze()

        optimizer.zero_grad()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss
    return total_loss
    

def test(args, test_loader, stock_net, criterion):
    predictions = []
    ground_truths = []
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            features, y = batch
            y = y.float().to(device)

            high_low_diffs = features.high_low_diff
            close_open_diffs = features.close_open_diff
            MA_7d = features.MAs.MA_7d
            MA_14d = features.MAs.MA_14d
            MA_21d = features.MAs.MA_21d
            std = features.std
            close = features.close

            input = torch.stack((high_low_diffs, close_open_diffs, std, MA_7d, MA_14d, MA_21d)).float().to(device)
            input = torch.transpose(input, 0, 1)
            output = stock_net(input).squeeze()

            total_loss += criterion(output, y)

            for prediction in output:
                predictions.append(prediction.item())
            for truth in y:
                ground_truths.append(truth.to('cpu'))

    x = [i for i in range(len(predictions))]
    plt.plot(x, predictions, color='blue', label='pred')
    plt.plot(x, ground_truths, color='red', label='ground truth')
    plt.legend(loc='best')
    plt.ylabel('price')
    plt.xlabel('time')
    plt.savefig('result.png')
    plt.clf()

    return total_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # stock related arguments
    parser.add_argument('--ticker', '-t', type=str, help='The ticker of company', default='NKE')
    parser.add_argument('--interval', '-i', type=str, help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='1d')
    parser.add_argument('--MA_intervals', '-m', nargs='+',help='list of intervals of moving average(unit: day)', default=[7, 14, 21])
    parser.add_argument('--std_interval', '-s', help='interval of std(unit: day)', type=int, default=7)
    parser.add_argument('--download','-d', action='store_true')
    parser.add_argument('--drop_head', help='drop the first drop_head rows of the csv file, since there are no value of MA and std for the first couple of days', type=int, default=21)
    parser.add_argument('--training_start', type=str, default='2011-01-01')
    parser.add_argument('--training_end', type=str, default='2020-12-31')
    parser.add_argument('--test_start', type=str, default='2021-01-01')
    parser.add_argument('--test_end', type=str, default='2023-02-16')


    # training related arguments
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    parser.add_argument('--in_features', type=int, default=6)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    args.MA_intervals = [int(interval) for interval in args.MA_intervals]

    main(args)