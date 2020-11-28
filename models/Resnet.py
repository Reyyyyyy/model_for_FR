import torch
import torch.nn as nn
import numpy as np

#New techs called label smoothing and loss scaling in https://ngc.nvidia.com/catalog/resources/nvidia:resnet_50_v1_5_for_pytorch

class Residual_Block(nn.Module):
    """Bottleneck"""
    def __init__(self,input_channels,output_channels,stride=1,he=True):
        super().__init__()

        self.input_channels  = input_channels
        self.output_channels = output_channels
        
        self.belly = nn.Sequential(
            nn.Conv2d(input_channels,64,kernel_size=1,stride=stride),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,64,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,output_channels,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels,kernel_size=3,padding=1,stride=stride**3),
            nn.BatchNorm2d(output_channels),
            )

        #对于deep network，采用He初始化
        if he == True:
            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                    layer.weight.data.normal_(0, np.sqrt(2. / n))
            
    def forward(self,x):
        identity = x
        out = self.belly(x)
        if self.input_channels != self.output_channels:
            identity = self.downsample(x)

        out = out + identity
        
        return nn.ReLU(True)(out)

class ResNet(nn.Module):
    def __init__(self,input_dim,output_dim,he=True):
        super().__init__()
        block = Residual_Block
        
        self.head = nn.Sequential(
            nn.Conv2d(input_dim,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        self.layer1 = self.make_layer(block, 64, 64, 3, 1,he)
        self.layer2 = self.make_layer(block, 64, 128, 3, 1,he)
        self.layer3 = self.make_layer(block, 128, 256, 6, 2,he)
        self.layer4 = self.make_layer(block, 256, 512, 3, 1,he)
        
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))# [batch_size, channels, ?, ?] -> [batch_size, channels, 1, 1]
        self.dropout = nn.Dropout()
        self.tail    = nn.Linear(512 , output_dim)
            

    def make_layer(self,block,input_channels,output_channels,num_blocks,stride,he):
        layers = []
        layers.append(block(input_channels, output_channels, stride))
        for layer in range(1,num_blocks):
            layers.append(block(output_channels,output_channels,1,he=he))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.squeeze(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.tail(x)
        return x    

if __name__ == '__main__':

    net = ResNet(input_dim=3,output_dim=2)
    test_img = np.random.normal(loc=0,scale=100,size=(1,3,224,224))
    test_tensor = torch.autograd.Variable(torch.Tensor(test_img))
    test_res = net(test_tensor)
    print(test_res)



        
        
    
