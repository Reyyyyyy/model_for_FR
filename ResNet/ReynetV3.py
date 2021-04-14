import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt

class time_based_model(nn.Module):
    """Three channels,Three LSTMs"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.processing = nn.LSTM(input_dim,
                                  hidden_size=100,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=True,    # 将输出变成[batch,seq,feature]
                                  bidirectional=False  # 设置成False是因为图片从左（或右）往右（或左）和从（或下）上往下（或上）看并没有本质区别
                                  )
        self.out = nn.Linear(100, output_dim)

    def forward(self, x):
        outs, _ = self.processing(x)
        out = outs[:, -1, :]
        return self.out(out)

class Residual_block(nn.Module):
    """identity_block"""

    def __init__(self, input_channels):
        super().__init__()
        self.processing = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, input_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(input_channels),
        )
        self.tail = nn.PReLU()

    def forward(self, x):
        out = self.processing(x)
        return self.tail(x + out)

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        block = Residual_block
        self.processing = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=7, padding=2, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),
            block(64),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),
            block(128),

            nn.Conv2d(128, 192, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(192),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),
            block(192),

            nn.Conv2d(192, 256, kernel_size=1, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.15),
            block(256),
        )
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.PReLU(),
            nn.Dropout(0.15),
            nn.Linear(2048, output_dim)
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, np.sqrt(2. / n))  # 均值为0，方差为np.sqrt(2. / n)
            elif isinstance(layer, nn.Linear):
                n = layer.in_features
                layer.weight.data.normal_(0, np.sqrt(2. / n))  # 均值为0，方差为np.sqrt(2. / n)

    def forward(self, x):
        x = self.processing(x)
        x = self.squeeze(x)
        x = x.reshape(x.shape[0], -1)
        return self.out(x)


class ReyNet(nn.Module):
    """Rey Mysterio!!!
    input_shape=(H,W,C)"""

    def __init__(self, input_shape, output_dim):
        super().__init__()
        H, W, C = input_shape
        self.time_based = time_based_model(W, output_dim)
        self.space_based = ResNet(C, output_dim)

    def forward(self, x):
        rs = x[:, 0, :, :]
        gs = x[:, 1, :, :]
        bs = x[:, 2, :, :]
        time_res = (self.time_based(rs) + self.time_based(gs) + self.time_based(bs)) / 3
        space_res = self.space_based(x)
        # return (time_res + space_res) / 2
        # return time_res
        return space_res

if __name__ == '__main__':
    cnn_net = ResNet(3, 10)
    test_img = np.random.normal(loc=0, scale=100, size=(3, 3, 32, 32))
    test_img = torch.autograd.Variable(torch.Tensor(test_img))
    res_1 = cnn_net(test_img)
    print(res_1)

    lstm_net = time_based_model(32, 10)
    test_seq = np.random.normal(loc=0, scale=100, size=(3, 32, 32))
    test_seq = torch.autograd.Variable(torch.Tensor(test_seq))
    res_2 = lstm_net(test_seq)
    print(res_2)

    rey_net = ReyNet((32, 32, 3), 10)
    res_3 = rey_net(test_img)
    print(res_3)




