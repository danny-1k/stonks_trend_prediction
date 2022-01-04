import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class Model:
    def save_model(self):
        torch.save(self.state_dict(),os.path.join('checkpoints',f'{type(self).__name__}.pt'))

    def load_model(self):
        self.load_state_dict(torch.load(os.path.join('checkpoints',f'{type(self).__name__}.pt')))


    def train_on(self,trainloader,testloader,optimizer,lossfn,scheduler=None,device='cuda' if torch.cuda.is_available() else 'cpu',epochs=5,plot_accuracy=False):
        
        if type(self).__name__ not in os.listdir('plots'):
            os.makedirs(f'plots/{type(self).__name__}')

        self.to(device)
        
        train_loss_over_time = []
        test_loss_over_time = []

        if plot_accuracy:
            accuracy_over_time = []

        last_loss = float('inf')

        for epoch in tqdm(range(epochs)):
            self.train()

            batch_train_loss = []
            batch_test_loss = []

            for x,y in trainloader:

                x = x.to(device)

                y = y.to(device).view(-1).long()

                p = self.__call__(x)

                loss = lossfn(p,y)
                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

                batch_train_loss.append(loss.item())


            if scheduler:
                scheduler.step()

            
            self.eval()

            with torch.no_grad():
                if plot_accuracy:
                    acc = 0

                for x,y in testloader:

                    x = x.to(device)

                    y = y.to(device).view(-1).long()

                    p = self.__call__(x)

                    if plot_accuracy:
                        acc+=(p.argmax(1)==y).sum()

                    loss = lossfn(p,y)

                    batch_test_loss.append(loss)

                if plot_accuracy:
                    acc = acc/len(testloader.dataset)


            train_loss = sum(batch_train_loss)/len(batch_train_loss)
            test_loss = sum(batch_test_loss)/len(batch_test_loss)

            train_loss_over_time.append(train_loss)
            test_loss_over_time.append(test_loss)
            
            if plot_accuracy:
                accuracy_over_time.append(acc)

            if test_loss_over_time[-1] < last_loss:
                self.save_model()
                last_loss = test_loss_over_time[-1]

            plt.plot(train_loss_over_time,label='Train loss')
            plt.plot(test_loss_over_time,label='Test loss')

            plt.legend()

            plt.savefig(os.path.join(f'plots/{type(self).__name__}','loss.png'))
            plt.close('all')

            if plot_accuracy:
                plt.plot(accuracy_over_time,label='Accuracy')
                plt.legend()

                plt.savefig(os.path.join(f'plots/{type(self).__name__}','accuracy.png'))
                plt.close('all')


class Conv(Model,nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2,out_channels=5,kernel_size=(2,4),padding=1)
        self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=(2,4),padding=1)
        self.conv3 = nn.Conv2d(in_channels=10,out_channels=15,kernel_size=(2,2),padding=1)
        self.maxpool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(90,2)

        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))        
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)

        return x