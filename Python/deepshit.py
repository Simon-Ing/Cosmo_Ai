
import torch
import torch.nn as nn
import torch.nn.functional as func


class ConvNet(nn.Module):
    def __init__(self, channels):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 6, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x))).view(-1, 16 * 4 * 4)
        x = self.fc3(func.relu(self.fc2(func.relu((self.fc1(x))))))
        return x

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.l3(self.relu(self.l2(self.relu(self.l1(x)))))