import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.nn.functional as func


def test_network(loader, model_, criterion_, device, print_results=False):
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            params = params.to(device)
            output = model_(images)
            loss_ = criterion_(output, params)
            total_loss += loss_
            n_batches += 1
            if print_results:
                for i, param in enumerate(params):
                    niceoutput = [round(n, 4) for n in output[i].tolist()]
                    niceparam = [round(n, 4) for n in param.tolist()]
                    print(f"{f'Correct: {niceparam}' : <50} {f'Output: {niceoutput}' : ^50}")
        return total_loss / n_batches


class CosmoDatasetPng(ImageFolder):
    def __init__(self, root):
        super(CosmoDatasetPng, self).__init__(root, transform=transforms.ToTensor())
        self.targets = torch.tensor([[int(a), int(b), int(c), int(d), int(e)] for (a, b, c, d, e) in [t[0].lstrip(
            root + "/images/").rstrip(".png").split(",") for t in self.imgs]], dtype=torch.float)

    def __getitem__(self, item):
        path, _ = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample[0].view(1, sample.shape[1], sample.shape[2]), self.targets[item]


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 32)
        x = self.fc1(x)
        return x

class ConvNet4(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 512)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNet5(nn.Module):
    def __init__(self):
        super(ConvNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        print(x.shape)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training parameters
num_epochs = 10
batch_size = 10
learning_rate = 0.001

if torch.cuda.is_available():
    print("Running Cuda!")
    device = torch.device("cuda")
else:
    print("Running on cpu")
    device = torch.device("cpu")

# Initialize your network, loss function, optimizer and scheduler
model = ConvNet5().to(device)
#model.load_state_dict(torch.load("Models/16"))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load a dataset for training and one for verificationtrain_dataset = CosmoDatasetPng("data_1000")
train_dataset = CosmoDatasetPng("medium")
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = CosmoDatasetPng("test")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

# Test network prior to training
loss = test_network(test_loader, model, criterion, device)
print(f'\nAverage loss over test data before training: {loss}\n')

print(f'Start training, num_epochs: {num_epochs}, batch size: {batch_size}, lr: {learning_rate}, \
        train samples: {len(train_dataset.imgs)} test samples: {len(test_dataset.imgs)}')

# Training loop
try:
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

        # Test network for each epoch
        loss = test_network(test_loader, model, criterion, device, print_results=False)
        print(f"Epoch: {epoch+1}, Loss: {loss}")

except KeyboardInterrupt:
    print("Training aborted by keyboard interrupt.")
except TypeError:
    print("Training aborted by keyboard interrupt.")


# Test network after training
loss = test_network(test_loader, model, criterion, device, print_results=False)
message = f'\nLoss: {loss}\nEpochs: {num_epochs}\nbatch_size: {batch_size}\nlearning_rate: {learning_rate}\ntrain samples: {len(train_dataset.imgs)}\n'

print(message)
print()

path = "Models/16"
torch.save(model.state_dict(), path)
