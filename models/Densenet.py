import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class dense_conv(nn.Module):
    """Conv -> BatchNorm -> ReLU"""
    def __init__(self,input_channels,output_channels,**kwargs):
        super().__init__()
        self.box = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(True),
            nn.Conv2d(input_channels,128,kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128,out_channels=output_channels,**kwargs)
            )
    def forward(self,x):
        return self.box(x)

class transition_layers(nn.Module):
    """BatchNorm -> ReLU -> 1*1 Conv -> 2*2 Average Pooling"""
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.box = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(True),
            nn.Conv2d(input_channels,output_channels,kernel_size=1),
            nn.AvgPool2d(kernel_size=2,stride=2)
            )
    def forward(self,x):
        return self.box(x)

class dense_block(nn.Module):
    """4 layers dense block"""
    def __init__(self,input_channels,growth_rate=32,dropout_rate=0.0,training=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.training = training
        conv = dense_conv

        self.dense_layer1 = conv(input_channels,growth_rate,kernel_size=3,padding=1)
        self.dense_layer2 = conv(input_channels+1*growth_rate,growth_rate,kernel_size=3,padding=1)
        self.dense_layer3 = conv(input_channels+2*growth_rate,growth_rate,kernel_size=3,padding=1)
        self.dense_layer4 = conv(input_channels+3*growth_rate,growth_rate,kernel_size=3,padding=1)

    def forward(self,x):
        x1 = self.dense_layer1(x)
        x1 = torch.cat([x,x1],1) #axis = 1 (channels axis)

        x2 = self.dense_layer2(x1)
        x2 = torch.cat([x1,x2],1)

        x3 = self.dense_layer3(x2)
        x3 = torch.cat([x2,x3],1)

        x4 = self.dense_layer4(x3)
        x4 = torch.cat([x3,x4],1)

        out = x4
        
        if self.dropout_rate > 0:
            out = F.dropout(out,p=self.dropout_rate,training=self.training)

        return out                 

class DenseNet(nn.Module):
    """input_shape = (height,width,channels)"""
    def __init__(self,input_shape,output_dim):
        super().__init__()
        self.mode = None
        input_height,input_width,input_channels = input_shape
        transition = transition_layers    
        dense = dense_block

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels,64,kernel_size=7,padding=3,stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
            )
        self.layer2 = dense(64,dropout_rate=0.5)
        self.layer3 = transition(192,256)
        self.layer4 = dense(256,dropout_rate=0.5)
        self.layer5 = transition(384,512)
        self.layer6 = dense(512,dropout_rate=0.5)
        self.layer7 = nn.AdaptiveAvgPool2d((1, 1)) # For 224*224 img , it is  14*14 -> 1*1
        self.layer8 = nn.Linear(640,output_dim)  

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = torch.flatten(x,1)
        return self.layer8(x)
            
if __name__ == '__main__':
    
    net = DenseNet((224,224,3),2)
    test_img = np.random.normal(loc=0,scale=100,size=(1,3,224,224))
    test_tensor = torch.autograd.Variable(torch.Tensor(test_img))
    test_res = net(test_tensor)
    print(test_res)







