# importing ResNet18 model and fitting to the 10 output classes of CIFAR 10

import torch.nn as nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.model(x)
        return x