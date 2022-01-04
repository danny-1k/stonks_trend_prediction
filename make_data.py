import os

import argparse

from data_helper import Stock,Combined

from tqdm import tqdm


time_steps = 16 # number of timesteps required to make prediction
pred_window = 5 #number of timesteps into the future to predict
train_ratio = .8 # train test split

def get_and_save_data():
    tickers = [ticker\
                for ticker in open('tickers.txt').read().splitlines()\
                if f'{ticker}.csv' in os.listdir(f'data/unprocessed')
            ]
    
    stock_list = [Stock(ticker,
                        seq_len=time_steps,
                        pred_window=pred_window,
                        train_ratio=train_ratio)\

                    for ticker in tqdm(tickers)    
                ]

    combined = Combined(stock_list)
    combined.save()


if __name__ == '__main__':
    get_and_save_data()