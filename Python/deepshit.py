
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset
import json
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
import os
import platform



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
    def __init__(self, root):
        super(CosmoDatasetPng, self).__init__(root, transform=transforms.ToTensor())
        if (platform.system() == 'Windows'):
            self.targets = torch.tensor([[int(a), int(b), int(c)] for (a, b, c) in [t[0].lstrip(root + "\\images\\").rstrip(".png").split(",") for t in self.imgs]], dtype=torch.float)
        else:
            self.targets = torch.tensor([[int(a), int(b), int(c)] for (a, b, c) in [t[0].lstrip(root + "/images/").rstrip(".png").split(",") for t in self.imgs]], dtype=torch.float)
    def __getitem__(self, item):
        path, _ = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample[0].view(1, sample.shape[1], sample.shape[2]), self.targets[item]




class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 7, (5, 5))
        self.conv3 = nn.Conv2d(7, 10, (5, 5))
        self.conv4 = nn.Conv2d(10, 10, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, 3)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))#.view(-1, 324)
        x = self.pool(func.relu(self.conv4(x)))#.view(-1, 144)
        x = self.pool(func.relu(self.conv4(x)))#.view(-1, 144)
        x = self.pool(func.relu(self.conv4(x))).view(-1, 40)
        #x = self.pool2(func.relu(self.conv5(x))).view(-1, 176)
        #print(x.shape)
        x = self.fc2(func.relu((self.fc1(x))))
        return x
