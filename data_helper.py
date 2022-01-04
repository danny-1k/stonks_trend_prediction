import os
import numpy as np
from numpy.core.defchararray import array
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 

class Stock:
    def __init__(self,ticker,seq_len,pred_window=10,train_ratio=.8,):
        self.ticker = ticker
        self.seq_len = seq_len
        self.pred_window = pred_window
        self.train_ratio = train_ratio

        assert (seq_len%2)==0, 'seq_len must be divisible by 2'
        assert seq_len>=pred_window ,'pred_window must be less than seq_len'

        ticker_df = pd.read_csv(f'data/unprocessed/{ticker}.csv')
        ticker_df.dropna(inplace=True)
        close = ticker_df['Close']
        volume = ticker_df['Volume']

        self.trainX,self.trainY,self.testX,self.testY = self.prepare_data(close,volume)


    def preprocess(self,x):

        x = [x[i:][:self.seq_len] for i in range(len(x)-self.seq_len)]
        x = [(i-i.min())/(i.max()-i.min()) for i in x]
        x = [np.array(i).reshape(2,-1) for i in x]

        return x

    def combine_close_volume(self,close,volume):
        x = [np.array([v,c]) for v,c in zip(close,volume)]
        return x


    def prepare_data(self,close,volume):
        close_prep = self.preprocess(close)
        volume_prep = self.preprocess(volume) #(len(x)-seq_len,2,seq_len/2)
        X = self.combine_close_volume(close_prep,volume_prep)
        Y = [1 if (close[i+self.pred_window]>=close[i]) else 0 for i in range(len(close)-self.seq_len)]

        
        data = list(zip(X,Y))
        
        np.random.shuffle(data)

        num_train = int(self.train_ratio*len(data))

        train = data[:num_train]
        test = data[num_train:]

        train_x,train_y = zip(*train)
        test_x , test_y = zip(*test)

        return train_x,train_y,test_x,test_y


class Combined:
    def __init__(self,stock_list=None,dir='data/processed'):
        self.dir = dir
        self.stock_list = stock_list


    def save(self):

        train_X = []
        train_Y = []

        test_X = []
        test_Y = []

        for stock in tqdm(self.stock_list):

            train_X+=stock.trainX
            train_Y+=stock.trainY

            test_X+=stock.testX
            test_Y+=stock.testY

        print('saving...')

        train = list(zip(train_X,train_Y))
        test = list(zip(test_X,test_Y))

        np.random.shuffle(train)
        np.random.shuffle(test)

        train_X,train_Y = zip(*train)
        test_X,test_Y = zip(*test)

        np.save(os.path.join(self.dir,'trainX'),train_X)
        np.save(os.path.join(self.dir,'trainY'),train_Y)

        np.save(os.path.join(self.dir,'testX'),test_X)
        np.save(os.path.join(self.dir,'testY'),test_Y)