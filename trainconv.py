import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Conv

from data import StockData

train = StockData(train=True)
test = StockData(train=False)

trainloader = DataLoader(train,batch_size=64,shuffle=True)
testloader = DataLoader(test,batch_size=64,shuffle=True)

net = Conv()

optimizer = optim.Adam(net.parameters(),lr=1e-5)

net.train_on(trainloader=trainloader,
            testloader=testloader,
            optimizer=optimizer,
            lossfn=nn.CrossEntropyLoss(),
            epochs=30,
            plot_accuracy=True,
)