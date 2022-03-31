import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCnn(nn.Module):
    def __init__(self, args=None):
        super(SmallCnn, self).__init__()
        self.classification = args.classification

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
        if self.classification:
            self.fc = nn.Linear(512 * 4 * 4, 101)
        else:
            self.fc = nn.Linear(512 * 4 * 4, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if self.classification:
            return x
        else:
            return x.squeeze()