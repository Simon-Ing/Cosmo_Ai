#! /usrbin/env python3

"""
Test Script for the CosmoAI model.
"""

from torch.utils.data import DataLoader
from Dataset import CosmoDataset
from Network import Inception3
from aux import test_network
import torch.nn as nn
import torch
import time
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast

# Training parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# checkpoints - set to true when you want new data points
load_checkpoint = False
checkpoint_path = "Models/autosave/autosave_epoch40"

device = torch.device("cpu")
# device = torch.device("cuda")

# Initialize your network, loss function, optimizer and scheduler
model = Inception3(num_outputs=8).to(device)
# model = torch.nn.DataParallel(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)


# Load a dataset for training and one for verification
train_dataset = CosmoDataset("train.csv")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CosmoDataset("test.csv")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
n_train_samples = len(train_dataset)
n_test_samples = len(test_dataset)
img_size = train_dataset[0][0].shape

# Load from checkpoint if desired:
if (load_checkpoint):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
        
# Test network prior to training
loss = test_network(test_loader, model, criterion, device)
print(f'\nAverage loss over test data before training: {loss}\n')

timer = time.time()
print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, \
        train samples: {n_train_samples} test samples: {n_test_samples} img size: {img_size}')

scaler = torch.cuda.amp.GradScaler()

# Training loop
try:
    for epoch in tqdm(range(num_epochs), desc="Total"):
        model.train()
        for i, (images, params) in enumerate(tqdm(train_loader, desc='Epoch')):
            print( i )
            print( images )
            print( params )
            images = images.to(device)
            params = params.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            
            with autocast():
                output = model(images)
                loss = criterion(output, params)
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step(loss)
        # Test network for each epoch
        model.eval()
        loss = test_network(test_loader, model, criterion, device, print_results=False)
        print(f"\nEpoch: {epoch+1}, Loss: {loss} "
              +"lr: {optimizer.state_dict()['param_groups'][0]['lr']}, time: {time.time() - timer}\n")
        
        # Save checkpoint
        if (epoch % 20 == 0) and (epoch > 0):
            autosave_path = "Models/autosave/" + "autosave_epoch" + str(epoch)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, autosave_path)

except KeyboardInterrupt:
    print("Training aborted by keyboard interrupt.")
except TypeError:
    print("TypeError.")


# Test network after training
loss = test_network(test_loader, model, criterion, device, print_results=True)
print( f'Loss: {loss}\nEpochs: {num_epochs}\nbatch_size: {batch_size}\n' )
print( f'learning_rate: {learning_rate}\nn_train_samples: {n_train_samples}\nimg_size: {img_size}' )

# Save your model if you want to
torch.save(model.state_dict(), "save-model")
