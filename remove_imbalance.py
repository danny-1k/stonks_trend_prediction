import numpy as np
import random
from tqdm import tqdm

trainX = np.load('data/processed/trainX.npy',allow_pickle=True)
trainY = np.load('data/processed/trainY.npy',allow_pickle=True)

testX = np.load('data/processed/testX.npy',allow_pickle=True)
testY = np.load('data/processed/testY.npy',allow_pickle=True)



white_train = len(trainY[trainY==1])
black_train = len(trainY[trainY==0])
train_imbalance = 1 if white_train >black_train else 0
train_remove = max([white_train,black_train])-min([white_train,black_train])

white_test = len(testY[testY==1])
black_test = len(testY[testY==0])
test_imbalance = 1 if white_test >black_test else 0
test_remove = max([white_test,black_test])-min([white_test,black_test])


print('TRAIN :','1 :',white_train,'0 :',black_train,'imbalance :',train_remove)
print('TEST  :','1 :',white_test, '0 :',black_test, 'imbalance ', test_remove)


trainX = trainX.tolist()
trainY = trainY.tolist()

testX = testX.tolist()
testY = testY.tolist()



num = 0

if white_train == black_train:
    print('No imbalance in train dataset')
else:
    for i,el in tqdm(enumerate(trainY)):
        if num <= train_remove and el == train_imbalance:
                del trainX[i]
                del trainY[i]
                num+=1

    np.save('data/processed/trainX.npy',trainX)
    np.save('data/processed/trainY.npy',trainY)


num = 0

if white_test == black_test:
    print('No imbalance in test dataset')
else:
    for i,el in tqdm(enumerate(testY)):
        if num <= test_remove and el == test_imbalance:
            del testX[i]
            del testY[i]
            num+=1

    np.save('data/processed/testX.npy',testX)
    np.save('data/processed/testY.npy',testY)