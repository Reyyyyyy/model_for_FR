import torch
from torch import nn
import numpy as np

class VggNet(nn.Module):
    """关于input_shape，举个例子，对于mnist数据，input_shape = (28,28,1)"""   
    def __init__(self,input_shape,output_dim):
        super().__init__()
        
        input_height,input_width,input_depth = input_shape
        self.output_dim = output_dim
        
        self.layer1 = nn.Sequential(nn.Conv2d(input_depth,64,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(64,64,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2,2)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(128,128,kernel_size=3,padding=1),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2,2)
                                   )
        self.layer3 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(256,256,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(256,256,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2,2,),
                                    )
        self.layer4 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2,2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(512,512,kernel_size=3,padding=1),
                                    nn.ReLU(True),
                                    nn.MaxPool2d(2,2)
                                    )
        self.layer6 = nn.AdaptiveAvgPool2d((1,1))
        self.layer7 = nn.Sequential(nn.Linear(512,1024),
                                    nn.ReLU(True),
                                    nn.Dropout(),
                                    nn.Linear(1024,self.output_dim)
                                    )
        #initialize_weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer6(x)
        x = x.view(x.shape[0],-1)#相当于flatten
    
        return self.layer7(x)
    
if __name__ == '__main__':                                                      
        
    net = VggNet((224,224,3),2)
    test_img = np.random.normal(loc=0,scale=100,size=(1,3,224,224))
    test_tensor = torch.autograd.Variable(torch.Tensor(test_img))
    test_res = net(test_tensor)
    print(test_res)





        
                                              
    
