
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import deepshit
import time

timer = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 1000
batch_size = 10  # 100: 98.94
learning_rate = 0.0002

train_dataset = deepshit.CosmoDataset(path='train_dataset.json')
test_dataset = deepshit.CosmoDataset(path='test_dataset.json')

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = deepshit.ConvNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, params) in enumerate(train_loader):
        images = images.to(device)
        params = params.to(device)
        # Forward pass
        output = model(images)
        loss = criterion(output, params)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Output: {output[0].tolist()}epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps} loss: {loss.item():.10f} time: {(time.time() - timer)}')


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        params = params.to(device)
        output = model(images)
        loss = criterion(output, labels)
        print(f'Loss: {loss}')


