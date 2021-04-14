import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#记得要做归一化！！！！

def load_data(normalization=True):
    test_adress  = 'G:/lab/1/datasets/ECM/dataset/new_arange_dataset/'
    test_xs  = np.load(test_adress+'test_batches.npy',allow_pickle=True)
    test_ys  = np.load(test_adress+'test_labels.npy',allow_pickle=True)
    test_xs  = np.transpose(test_xs,[0,3,1,2])
    test_ys  = np.argmax(test_ys,axis=1)    
    if normalization:
        test_xs  = test_xs/255.
    return (test_xs,test_ys)

def get_evaluation_parameters():
    #test_batch_size
    batch_size = 10
    #evaluation params
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    #load data
    (x_test,y_test) = load_data(normalization=True)
    #arange data
    with torch.no_grad():
        x_test  = torch.Tensor(x_test).cuda()
        y_test  = torch.Tensor(y_test).cuda()
    test_data   = torch.utils.data.TensorDataset(x_test,y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=False)
    #evaluate
    print('Testing\n----------------------')
    model.eval()
    for step,(batch_x,batch_y) in enumerate(test_loader):
        pred = np.reshape(np.argmax(np.array((model(batch_x)).tolist()),axis=1),(1,-1))
        real = np.reshape(np.array(batch_y.tolist()),(1,-1))
        TP   += np.sum(np.logical_and(np.equal(real,0),np.equal(pred,0)))
        FP   += np.sum(np.logical_and(np.equal(real,1),np.equal(pred,0)))
        TN   += np.sum(np.logical_and(np.equal(real,1),np.equal(pred,1)))
        FN   += np.sum(np.logical_and(np.equal(real,0),np.equal(pred,1)))
    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F_score   = (2*Precision*Recall)/(Precision+Recall)
    print('TP:',TP)
    print('TN:',TN)
    print('FP:',FP)    
    print('FN:',FN)
    print('----------------------')
    print('Accuracy： {:.4f}'.format(Accuracy))
    print('Precision：{:.4f}'.format(Precision))
    print('Recall：   {:.4f}'.format(Recall))
    print('F_score：  {:.4f}'.format(F_score))
    params = dict(Accuracy=Accuracy,Precision=Precision,Recall=Recall,F_score=F_score)
    return params

if __name__ == '__main__':
    model = torch.load(r'my_model')
    model.eval()#不使用dropout和BN，在预测和评估模型时使用
    params = get_evaluation_parameters()
