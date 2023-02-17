from torch import nn

class StockNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()


        self.net =  nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, out_features),
        )

    def forward(self, x):
        return self.net(x)