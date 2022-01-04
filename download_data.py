import os
import argparse

from tqdm import tqdm
import datetime

import pandas_datareader.data as web


parser = argparse.ArgumentParser(description='Program to download stock data from S&P 100')
parser.add_argument('--start', default='(1980,1,1)',
                    help="Start Date (year,month,day)")
parser.add_argument('--end', default=datetime.datetime.now().strftime(
    '(%Y,%m,%d)'), help="End Date (year,month,day)")

parser.add_argument('--window', default=7, help="Time window used to determine the change in price")

args = parser.parse_args()

start = datetime.date(*eval(args.start)).strftime('%Y-%m-%d')
end = datetime.date(*eval(args.end)).strftime('%Y-%m-%d')

tickers = open('tickers.txt', 'r').read().split('\n')

num = 0
for tick in tqdm(tickers):
    try:
        if f'{tick}.csv' not in os.listdir('data/unprocessed'):
            df = web.DataReader(name=tick, data_source='yahoo', start=start, end=end)

            if f'{tick}.csv' not in os.listdir('data/unprocessed'):
                df[['Open','High','Low','Close','Volume']].to_csv(f'data/unprocessed/{tick}.csv')

        num+=1

    except:
        continue 

print(f'Downloaded {num} of {len(tickers)} .')
