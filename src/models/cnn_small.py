import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModelSmall(nn.Module):
    def __init__(self, args=None):
        super(MyModelSmall, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(512 * 4 * 4, 101)

    def forward(self, x):
        x = self.encoder(x)
        #import pdb; pdb.set_trace()
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x