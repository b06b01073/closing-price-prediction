import argparse
import utils
from torch.utils.data import DataLoader
import torch
import model
from torch import optim
from torch import nn 
import numpy as np

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    train_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.training_start, end=args.training_end, train_set=True)
    test_set = utils.get_dataset(ticker=args.ticker, interval=args.interval, MA_intervals=args.MA_intervals, std_interval=args.std_interval, download=args.download, drop_head=args.drop_head, start=args.test_start, end=args.test_end, train_set=False)

    print(f'len of train_set: {len(train_set)}')
    print(f'len of test_set: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)


    stock_net = model.StockNet(in_features=args.in_features, out_features=args.out_features).to(device)
    optimizer = optim.RMSprop(stock_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss(reduction='sum')

    best_train_loss = float('inf')
    best_test_loss = float('inf')
    best_out = None
    count = 0
    for epoch in range(args.epochs):
        train_loss = train(args, train_loader, stock_net, optimizer, criterion)
        test_loss, output = test(args, test_loader, stock_net, criterion)

        count += 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(stock_net.state_dict(), utils.get_path(f'./model_params/{args.ticker}.pth'))

            features = test_set.fetch_data(-2)
            features = [features.open, features.high, features.low, features.close, features.high_low_diff, features.close_open_diff, features.std, features.MAs.MA_3d, features.MAs.MA_7d, features.MAs.MA_14d, features.MAs.MA_21d]

            features = torch.from_numpy(np.stack(features, axis=0)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                best_out = stock_net(features).item()

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            count = 0

        if count >= args.early_stop:
            break

        
        error_log(train_loss / len(train_loader), test_loss / len(test_loader), epoch, best_out)

def error_log(train_loss, test_loss, epoch, best_out):
    print(f'epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}, best_out: {best_out}')

def train(args, train_loader, stock_net, optimizer, criterion):
    total_loss = 0
    for batch in train_loader:
        features, y = batch
        y = y.float().to(device)


        input = torch.stack((features.open, features.high, features.low, features.close, features.high_low_diff, features.close_open_diff, features.std, features.MAs.MA_3d, features.MAs.MA_7d, features.MAs.MA_14d, features.MAs.MA_21d)).float().to(device)
        input = torch.transpose(input, 0, 1)

        output = stock_net(input).flatten()

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
        for idx, batch in enumerate(test_loader):
            features, y = batch
            y = y.float().to(device)

            input = torch.stack((features.open, features.high, features.low, features.close,  features.high_low_diff, features.close_open_diff, features.std, features.MAs.MA_3d, features.MAs.MA_7d, features.MAs.MA_14d, features.MAs.MA_21d)).float().to(device)
            input = torch.transpose(input, 0, 1)
            output = stock_net(input).flatten()

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
    plt.xlabel(f'trading day since {args.test_start}')
    plt.title(args.ticker)
    plt.savefig(f'./result/{args.ticker}.png')
    plt.clf()

    return total_loss, output



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # stock related arguments
    parser.add_argument('--ticker', '-t', type=str, help='The ticker of company', default='NKE')
    parser.add_argument('--interval', '-i', type=str, help='https://github.com/ranaroussi/yfinance/wiki/Ticker', default='1d')
    parser.add_argument('--MA_intervals', '-m', nargs='+',help='list of intervals of moving average(unit: day)', default=[3, 7, 14, 21])
    parser.add_argument('--std_interval', '-s', help='interval of std(unit: day)', type=int, default=7)
    parser.add_argument('--download','-d', action='store_true')
    parser.add_argument('--drop_head', help='drop the first drop_head rows of the csv file, since there are no value of MA and std for the first couple of days', type=int, default=21)
    parser.add_argument('--training_start', type=str, default='2011-01-01')
    parser.add_argument('--training_end', type=str, default='2021-12-31')
    parser.add_argument('--test_start', type=str, default='2022-01-01')
    parser.add_argument('--test_end', type=str, default='2023-02-17')


    # training related arguments
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=1000)
    parser.add_argument('--in_features', type=int, default=11)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=100)

    args = parser.parse_args()
    args.MA_intervals = [int(interval) for interval in args.MA_intervals]

    main(args)