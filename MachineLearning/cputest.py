#! /usrbin/env python3

"""
Test Script for the CosmoAI model.
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

# Training parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# Initialize your network, loss function, and optimizer 
model = Inception3(num_outputs=8)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load a dataset for training and one for verification
train_dataset = CosmoDataset("train.csv")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CosmoDataset("test.csv")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
n_train_samples = len(train_dataset)
n_test_samples = len(test_dataset)
img_size = train_dataset[0][0].shape

# Test network prior to training
loss = testCPU(test_loader, model, criterion)
print(f'\nAverage loss over test data before training: {loss}\n')

timer = time.time()
print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, \
        train samples: {n_train_samples} test samples: {n_test_samples} img size: {img_size}')

# Training loop
try:
    for epoch in range(num_epochs):
        for i, (images, params) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward + Backward + Optimiser
            output = model(images)
            loss = criterion(output, params)
            loss.backward()
            optimizer.step()
            print( f"Batch no. {epoch}-{i}: loss = {loss.item()}\n" )

        # Test network for each epoch
        model.eval()
        loss = testCPU(test_loader, model, criterion)
        print(f"\nEpoch: {epoch+1}, Loss: {loss} "
              +"lr: {optimizer.state_dict()['param_groups'][0]['lr']}, time: {time.time() - timer}\n")
        
except KeyboardInterrupt:
    print("Training aborted by keyboard interrupt.")
except TypeError:
    print("TypeError.")


# Test network after training
loss = testCPU(test_loader, model, criterion)
print( f'Loss: {loss}\nEpochs: {num_epochs}\nbatch_size: {batch_size}\n' )
print( f'learning_rate: {learning_rate}\nn_train_samples: {n_train_samples}\nimg_size: {img_size}' )

# Save your model if you want to
torch.save(model.state_dict(), "save-model")
