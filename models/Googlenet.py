import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import warnings

class BasicConv2d(nn.Module):
    """Conv -> BN -> ReLU"""
    def __init__(self, input_channels, output_channels, **kwargs):#**kwargs means needing to input a = b 
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class Inception_block(nn.Module):
    def __init__(self,input_channels,conv1x1,conv3x3_front,conv3x3_back,_conv3x3_front,_conv3x3_back,outpool_channels):
        super().__init__()
        
        conv_block = BasicConv2d# Conv -> BN -> ReLU
        
        self.branch1 = conv_block(input_channels,conv1x1,kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(input_channels,conv3x3_front,kernel_size=1),
            conv_block(conv3x3_front,conv3x3_back,kernel_size=3,padding=1)
            )
        self.branch3 = nn.Sequential(
            conv_block(input_channels,_conv3x3_front,kernel_size=1),
            conv_block(_conv3x3_front,_conv3x3_back,kernel_size=3,padding=1)
            )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            conv_block(input_channels,outpool_channels,kernel_size=1)
            )
        
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)        
        
        outputs = [branch1, branch2, branch3, branch4]#output_channels = conv1x1 + conv3x3_back +_conv3x3_back + outpool_channels
        
        return torch.cat(outputs, 1)#concat by channels
    
class GoogleNet(nn.Module):
    """input_shape，举例，对于cifar-10,input_shape=(32,32,3)"""
    def __init__(self,input_shape,output_dim):
        super().__init__()
        warnings.warn("Just a warning,nothing needs to be worried")
        
        input_height,input_width,input_depth = input_shape
        self.output_dim = output_dim

        conv_block = BasicConv2d# Conv -> BN -> ReLU
        inception_block = Inception_block
        
        self.layer1 = nn.Sequential(conv_block(input_depth,64,kernel_size=7,stride=2,padding=3),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True),
                                    conv_block(64,64,kernel_size=1),
                                    conv_block(64,192,kernel_size=3,padding=1),
                                    nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
                                    )
        self.layer2 = nn.Sequential(inception_block(192, 64, 96, 128, 16, 32, 32),
                                    inception_block(256, 128, 128, 192, 32, 96, 64),
                                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
                                    )
        self.layer3 = nn.Sequential(inception_block(480, 192, 96, 208, 16, 48, 64),
                                    inception_block(512, 160, 112, 224, 24, 64, 64),
                                    inception_block(512, 256, 160, 320, 32, 128, 128),
                                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
                                    )
        self.layer4 = nn.Sequential(inception_block(832, 256, 160, 320, 32, 128, 128),
                                    inception_block(832, 384, 192, 384, 48, 128, 128)
                                    )
        self.layer5 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Dropout(0.5)#abandon rate
                                    )
        self.layer6 = nn.Linear(1024, output_dim)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)

        return self.layer6(x)
    
if __name__ == '__main__':

    net = GoogleNet((224,224,3),2)
    test_img = np.random.normal(loc=0,scale=100,size=(1,3,224,224))
    test_tensor = torch.autograd.Variable(torch.Tensor(test_img))
    test_res = net(test_tensor)
    print(test_res)








        
