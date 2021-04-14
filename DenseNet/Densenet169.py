import numpy as np
import torch
from torchvision import models
from torch import nn
from matplotlib import pyplot as plt

class Densenet169(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        densenet = models.densenet169(pretrained=True)
        self.features_extractor = nn.Sequential(*list(densenet.children())[0])
        for layer in self.features_extractor:
            if layer._get_name() == '_Transition':
                layer.add_module('dropout2d', nn.Dropout2d(0.15))
        self.features_extractor.add_module('global average',nn.AdaptiveAvgPool2d((1,1)))
        self.fully_connected_layers = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(1664, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, num_classes)
        )
        for layer in self.fully_connected_layers.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        features = self.features_extractor(x)
        flatten_features = features.view(features.shape[0], -1)
        return self.fully_connected_layers(flatten_features)


if __name__ == '__main__':
    densenet = Densenet169(num_classes=10)
    test_img = np.random.normal(loc=0, scale=100, size=(2, 3, 32, 32))
    test_img = torch.autograd.Variable(torch.Tensor(test_img))
    res = densenet(test_img)
    print(res)

