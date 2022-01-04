import torch

import numpy as np

import pandas as pd

import datetime


class Sim:
    def __init__(self,tickers,initial_deposit=100,time_frame=('2020-01-01','2021-01-01'),log=True):
        self.tickers = tickers
        self.initial_deposit = initial_deposit
        self.bank = initial_deposit
        self.time_frame = time_frame
        self.log = log

        self.super_world = {ticker:pd.read_csv(f'data/unprocessed/{ticker}.csv') for ticker in tickers}
        
        self.world = {ticker:self.super_world[ticker].loc[(self.super_world[ticker]['Date']>=time_frame[0]) & (self.super_world[ticker]['Date']<=time_frame[1])] for ticker in tickers}

        self.current_day = self.world[tickers[0]]['Date'].iloc[0]
        self.day = 0

        self.num_shares_owned = {ticker:0 for ticker in tickers}

    def next_day(self):
        date = datetime.date(*[int(i) for i in self.current_day.split('-')])
        date+=datetime.timedelta(days=1)
        self.current_day = date.strftime('%Y-%m-%d')

        self.day+=1

        # if self.current_day == self.time_frame[-1]:
        #     print('End of Simulation')
        #     return True

        if self.day == len(self.world[self.tickers[0]])-1:
            print('End of simulation')
            return True

    def buy(self,ticker):
        # print(self.current_day)
        # print(self.world['Date'])
        # price = self.world.loc[self.world['Date']==self.current_day]['Close']

        price = self.world[ticker].iloc[self.day]['Close']

        if self.bank > price:
            if self.log:
                print('-'*40)
                print(f'Bought 1 Share of {ticker} stock @ {price}')
                print('-'*40)
                print()
            self.bank -= price
            self.num_shares_owned[ticker]+=1

    def sell(self,ticker):
        # print(self.current_day)
        # print(self.world['Date'])
        # price = self.world.loc[self.world['Date']==self.current_day]['Close']

        price = self.world[ticker].iloc[self.day]['Close']

        if self.num_shares_owned[ticker] >0:
            if self.log:
                print('-'*40)
                print(f'SOLD 1 Share of {ticker} stock @ {price}')
                print('-'*40)
                print()
            self.bank += price
            self.num_shares_owned[ticker] -=1

    def print_account(self):
        print()
        print('-'*40)
        print(f'Num of shares owned : {self.num_shares_owned}')
        print(f'Bank                : ${self.bank}')
        print(f'Initial deposit     : ${self.initial_deposit}')
        print(f'Number of days done : {self.day}')
        
        print('-'*40)
        print()


    def get_x(self,ticker):
        # print(self.world.iloc[self.day].index)
        idx = self.world[ticker].index[self.day]
        y = self.super_world[ticker]['Close'][idx+5] > self.super_world[ticker]['Close'][idx]

        close_30 = self.super_world[ticker]['Close'][:idx][-16:]
        volume_30 = self.super_world[ticker]['Volume'][:idx][-16:]

        close_30 = (close_30-close_30.min())/(close_30.max()-close_30.min()).tolist()
        volume_30 = (volume_30-volume_30.min())/(volume_30.max()-volume_30.min()).tolist()

        close_30 = np.array(close_30).reshape((2,-1))
        volume_30 = np.array(volume_30).reshape((2,-1))

        x = np.array([volume_30,close_30])

        x = torch.from_numpy(x).unsqueeze(0).float()

        return x,y

    def add_more_money(self,amount):
        self.bank+=amount
