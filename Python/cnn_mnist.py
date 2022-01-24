
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import deepshit
import time

timer = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
batch_size = 100  # 100: 98.94
learning_rate = 0.05

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = deepshit.ConvNet(channels=1).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        imgs = images.to(device)
        lbls = labels.to(device)
        # Forward pass
        output = model(imgs)
        loss = criterion(output, lbls)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps} loss: {loss.item():.10f} time: {(time.time() - timer)}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        imgs = images.to(device)
        lbls = labels.to(device)
        outputs = model(imgs)

        _, predicted = torch.max(outputs, 1)
        n_samples += lbls.size(0)
        n_correct += (predicted == lbls).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label.item() == pred.item():
                n_class_correct[label] += 1
            n_class_samples[label] += 1


    acc = 100 * n_correct / n_samples
    print(f'accuracy of network: {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'accuracy of {i}: {acc}%')


