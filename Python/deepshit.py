
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset
import json
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms



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
        ret1 = sample[0].view(1, sample.shape[1], sample.shape[2]).numpy()
        ret2 = self.targets[item]
        return ret1, ret2


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, (5, 5))
        self.conv2 = nn.Conv2d(3, 5, (5, 5))
        self.conv3 = nn.Conv2d(5, 7, (5, 5))
        self.conv4 = nn.Conv2d(7, 9, (5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(144, 20)
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        x = self.pool3(func.relu(self.conv1(x)))
        x = self.pool3(func.relu(self.conv2(x)))
        x = self.pool3(func.relu(self.conv3(x)))
        x = self.pool2(func.relu(self.conv4(x))).view(-1, 144)
        x = self.fc2(func.relu((self.fc1(x))))
        return x
