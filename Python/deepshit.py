import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset
import json
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))  # one input (colors), six outputs, kernel(filter) of size 5x5
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 16, (5, 5))
        self.conv4 = nn.Conv2d(16, 20, (5, 5))
        self.conv5 = nn.Conv2d(20, 20, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3920, 1000)  # fully connected (normal vanilla layer), inputs: 128, outputs:128
        self.fc2 = nn.Linear(1000, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 30)
        self.fc5 = nn.Linear(30, 3)

    def forward(self, x):

        x = self.conv1(x)
        x = func.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = func.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = func.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = func.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = func.relu(x)
        x = self.pool(x).view(-1, 3920)

        # After convolving and pooling we end up with a 1D array, we feed this into the neural net
        x = self.fc1(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = func.relu(x)
        x = self.fc3(x)  # We end up with three values (x position, Einstein Radius and source size
        x = func.relu(x)
        x = self.fc4(x)
        x = func.relu(x)
        x = self.fc5(x)
        x = func.relu(x)
        return x


class CosmoDatasetJson(Dataset):
    def __init__(self, path):
        n_samples = 100
        x = np.zeros((n_samples, 600*600))
        y = np.zeros((n_samples, 3))
        for i in range(n_samples):
            path = "data/cosmo_data/datapoint" + str(i + 900) + ".json"
            with open(path, "r") as infile:
                indata = json.load(infile)
            x[i] = (indata["image"])
            y[i, 0] = indata["actualPos"]
            y[i, 1] = indata["einsteinR"]
            y[i, 2] = indata["source_size"]
        self.x = torch.tensor(x, dtype=torch.float).view(-1, 1, 600, 600)
        self.y = torch.tensor(y, dtype=torch.float)
        self.n_samples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



class CosmoDatasetPng(ImageFolder):

    def __getitem__(self, item):
        path, _ = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        ret1 = sample[0].view(1, 600, 600).numpy()
        ret2 = self.targets[item]
        return ret1, ret2

        # x = self.conv1(x)
        # x = func.leaky_relu(x)  # non-linear activation function
        # x = self.pool(x)
        # x = self.conv2(x)
        # x = func.leaky_relu(x)
        # x = self.pool(x)
        # x = self.conv3(x)
        # x = func.leaky_relu(x)
        # x = self.pool(x)
        # x = self.conv4(x)
        # x = func.leaky_relu(x)
        # x = self.pool(x)
        # x = self.conv4(x)
        # x = func.leaky_relu(x)
        # x = self.pool(x)#.view(-1, 128)
        # x = self.conv4(x)
        # x = func.leaky_relu(x)
        # x = self.pool(x)#.view(-1, 128)