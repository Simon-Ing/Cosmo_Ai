
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset
import json

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))  # one input (colors), six outputs, kernel(filter) of size 5x5
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 64, (5, 5))
        self.conv4 = nn.Conv2d(64, 128, (5, 5))
        self.pool4 = nn.MaxPool2d(4, 4)  # Takes max value from each set of 4x4 pixels
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 128)  # fully connected (normal vanilla layer), inputs: 128, outputs:128
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)  # non-linear activation function
        x = self.pool4(x)
        x = self.conv2(x)
        x = func.relu(x)
        x = self.pool4(x)
        x = self.conv3(x)
        x = func.relu(x)
        x = self.pool4(x)
        x = self.conv4(x)
        x = func.relu(x)
        x = self.pool2(x).view(-1, 128)

        # After convolving and pooling we end up with a 1D array, we feed this into the neural net
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = func.relu(x)
        x = self.fc3(x)  # We end up with three values (x position, Einstein Radius and source size
        return x


class CosmoDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as infile:
            in_data = json.load(infile)
        self.x = torch.tensor(in_data["images"], dtype=torch.float)
        self.y = torch.tensor(in_data["params"], dtype=torch.float)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        ret_x = self.x[index]
        return torch.reshape(ret_x, (1, ret_x.shape[0], ret_x.shape[1])), self.y[index]

    def __len__(self):
        return self.n_samples
