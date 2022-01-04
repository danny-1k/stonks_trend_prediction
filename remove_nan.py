import numpy as np


trainX = np.load('data/processed/trainX.npy',allow_pickle=True)
trainY = np.load('data/processed/trainY.npy',allow_pickle=True)

testX = np.load('data/processed/testX.npy',allow_pickle=True)
testY = np.load('data/processed/testY.npy',allow_pickle=True)

trainX = trainX.tolist()
trainY = trainY.tolist()

testX = testX.tolist()
testY = testY.tolist()


didChangeTrain = False
didChangeTest = False

for i,el in enumerate(trainX):
    if np.isnan(np.array(el)).any():
        del trainX[i]
        del trainY[i]
        if not didChangeTrain:
            didChangeTrain = True

for i,el in enumerate(testX):
    if np.isnan(np.array(el)).any():
        del testX[i]
        del testY[i]
        if not didChangeTest:
            didChangeTest = True


if not didChangeTrain:
    print('There aren\'t any NaN values in the Train dataset')
else:

    np.save('data/processed/trainX.npy',trainX)
    np.save('data/processed/trainY.npy',trainY)

if not didChangeTest:
    print('There aren\'t any NaN values in the Test dataset')

else:
    np.save('data/processed/testX.npy',testX)
    np.save('data/processed/testY.npy',testY)