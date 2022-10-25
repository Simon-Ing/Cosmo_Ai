#! /usrbin/env python3

"""
The MLSystem class provides defaults for all the components of
a machine learning system.  The default implementation uses the CPU.
Subclasses should override key functions for more advanced systems.
"""

from torch.utils.data import DataLoader
from Dataset import CosmoDataset
from Network import Inception3
from aux import testCPU
import torch.nn as nn
import torch
import time
from tqdm import tqdm
from torch.cuda.amp import autocast

class MLSystem:
   def __init__(self,model=None,criterion=None,optimizer=None,nepoch=20):
        self.num_epochs = nepoch
        self.batch_size = 32
        self.learning_rate = 0.001

        # Initialize your network, loss function, and optimizer 
        if model == None:
           self.model = Inception3(num_outputs=8)
        if criterion == None:
           self.criterion = nn.MSELoss()
        if optimizer == None:
           self.optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=0.001, momentum=0.9)
   def loadtrainingdata(self,fn="train.csv"):
       train_dataset = CosmoDataset(fn)
       self.trainloader = DataLoader(dataset=train_dataset,
                                 batch_size=self.batch_size, shuffle=True)
       self.img_size = train_dataset[0][0].shape
       self.ntrain = len(train_dataset)
   def loadtestdata(self,fn="train.csv"):
       test_dataset = CosmoDataset("test.csv")
       self.testloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)
       self.ntest = len(test_dataset)

   def printparam(self):
     print(f'num_epochs: {self.num_epochs}, ' 
             + f'batch size: {self.batch_size}, lr: {self.learning_rate}\n' )
     print(f'img size: {self.img_size}')
     print(f'train samples: {self.ntrain} test samples: {self.ntest}\n' )
   def train(self):
     timer = time.time()
     print('Start training:')
     self.printparam()
     try:
         for epoch in range(self.num_epochs):
             self.model.train()
             for i, (images, params) in enumerate(self.trainloader):
                 self.optimizer.zero_grad()
                 
                 # Forward + Backward + Optimiser
                 output = self.model(images)
                 loss = self.criterion(output, params)
                 loss.backward()
                 self.optimizer.step()
                 print( f"Batch no. {epoch}-{i}: loss = {loss.item()}" )

             # Test network for each epoch
             self.model.eval()
             loss = testCPU(self.testloader, self.model, self.criterion)
             print(f"\nEpoch: {epoch+1}, Loss: {loss} "
                   +"lr: {optimizer.state_dict()['param_groups'][0]['lr']}, time: {time.time() - timer}\n")
        
     except KeyboardInterrupt:
         print("Training aborted by keyboard interrupt.")
     except TypeError:
         print("TypeError.")

   def getLoss(self):
         loss = testCPU(self.testloader, self.model, self.criterion)
   def savemodel(self,fn="save-model"):
         torch.save(self.model.state_dict(), fn)

if __name__ == "__main__":
    print( "MLSystem test script.\nConfiguring ... " )

    ml = MLSystem()
    ml.loadtrainingdata()
    ml.loadtestdata()

    print( "Pre-training test ..." )

    loss = ml.getLoss()
    print(f'\nAverage loss over test data before training: {loss}\n')

    print( "Training ..." )

    ml.train()

    print( "Post-training test ..." )
    loss = ml.getLoss()
    print( f'Loss: {loss}' )

    print( "Saving ..." )
    ml.savemodel()
