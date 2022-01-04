import torch
from torch.utils.data import DataLoader

import numpy as np

from data import StockData
from models import Conv

import matplotlib.pyplot as plt

import random

from sim import Sim

test = StockData(train=False)

net = Conv()
net.load_model()
net.eval()

simulation = Sim(tickers=['GOOG'],time_frame=('2020-01-01','2021-06-30'), initial_deposit=100000,log=False)

simulation.print_account()

same = []

while True:
    for ticker in simulation.tickers:

        x,y = simulation.get_x(ticker)

        pred = net(x).squeeze()

        action = pred.argmax()

        same.append((action ==y).item())

        if action == 1 : # Increase in the next 10 days; BUY
            simulation.buy(ticker)

        else: #SELL
            simulation.sell(ticker)


    if simulation.next_day():
        for ticker in simulation.tickers:
            if simulation.num_shares_owned[ticker]>0:
                print(f'REMAINING SHARES HELD IN {ticker} -> {simulation.num_shares_owned[ticker]}')
                print(f'SELLING ALL REMAINING SHARES in {ticker}')
        for ticker in simulation.tickers:
            if simulation.num_shares_owned[ticker]>0:
                for s in range(simulation.num_shares_owned[ticker]):
                    simulation.sell(ticker)

        simulation.print_account()
        print(sum(same)/len(same))
        break

    # simulation.add_more_money(500)

# for i in range(10):
#     x,y = test[i]
#     p = net(x.unsqueeze(0))
#     print(p.argmax(),y)