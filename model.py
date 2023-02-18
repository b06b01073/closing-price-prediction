from torch import nn
from torchsummary import summary
import torch

class StockNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()


        self.net =  nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    stock_net = StockNet(in_features=12, out_features=1).to('cuda')
    summary(stock_net, (11,))